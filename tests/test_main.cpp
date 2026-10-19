// gtest: tiny assert-based test runner (no googletest dependency).
// Register tests with TEST(name) { ... } and check with CHECK(cond).
// A failing CHECK records the failure and continues with the next test.
#include "cpu/bfs_cpu.hpp"
#include "cpu/pagerank_cpu.hpp"
#include "graph/csr.hpp"
#include "graph/generators.hpp"
#include "graph/reorder.hpp"
#include "graph/snap.hpp"
#include "graph/transform.hpp"
#include "metal/bfs_gpu.hpp"
#include "metal/context.hpp"
#include "metal/pagerank_gpu.hpp"
#include "metal/primitives.hpp"
#include "util/rng.hpp"

#include <cstdlib>
#include <unistd.h>
#include <zlib.h>

#include <omp.h>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

struct TestCase { std::string name; std::function<void()> fn; };
std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; }
int g_failures = 0;
bool g_current_failed = false;

struct Registrar {
    Registrar(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

} // namespace

#define TEST(name)                                              \
    static void test_##name();                                  \
    static Registrar reg_##name(#name, test_##name);            \
    static void test_##name()

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::printf("    CHECK failed: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
            g_current_failed = true;                                                  \
            return;                                                                   \
        }                                                                             \
    } while (0)

// Shared GPU context: compiling shaders per test would dominate test time.
static gpu::MetalContext& ctx() { static gpu::MetalContext c; return c; }

// ---------------------------------------------------------------- Phase 0

TEST(openmp_has_threads) {
    int team = 0;
#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    CHECK(team >= 1);
}

// Vector add at awkward sizes: 0, 1, and non-multiples of the threadgroup size.
TEST(metal_vector_add_sizes) {
    id<MTLComputePipelineState> pso = ctx().pipeline("micro", "vector_add");
    for (uint32_t n : {0u, 1u, 255u, 256u, 257u, 100003u}) {
        std::vector<float> a(n), b(n);
        for (uint32_t i = 0; i < n; ++i) { a[i] = (float)i; b[i] = 2.0f * i + 1; }
        id<MTLBuffer> A = ctx().buffer(a.data(), n * 4), B = ctx().buffer(b.data(), n * 4);
        id<MTLBuffer> C = ctx().buffer(n * 4);
        id<MTLCommandBuffer> cb = [ctx().queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:B offset:0 atIndex:1];
        [enc setBuffer:C offset:0 atIndex:2];
        [enc setBytes:&n length:4 atIndex:3];
        gpu::MetalContext::dispatch1D(enc, pso, n, 256);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* c = (const float*)C.contents;
        for (uint32_t i = 0; i < n; ++i) CHECK(c[i] == a[i] + b[i]);
    }
}


// ---------------------------------------------------------------- Phase 1

using graph::CSR;
using EdgeList = std::vector<std::pair<uint32_t, uint32_t>>;

// Scratch file path, unique per process; removed by the tests that create it.
static std::string tmpPath(const std::string& name) {
    const char* dir = std::getenv("TMPDIR");
    std::string d = dir ? dir : "/tmp";
    if (!d.empty() && d.back() != '/') d += '/';
    return d + "ggpu_test_" + std::to_string(getpid()) + "_" + name;
}

static bool hasEdge(const CSR& g, uint32_t u, uint32_t v) {
    return std::binary_search(g.begin(u), g.end(u), v);
}

TEST(generators_valid_and_shaped) {
    for (auto& [name, g] : graph::gen::suite()) {
        std::string err = graph::validate(g);
        if (!err.empty()) std::printf("    %s: %s\n", name.c_str(), err.c_str());
        CHECK(err.empty());
    }
    CSR p = graph::gen::path(1000);
    CHECK(p.n == 1000 && p.m == 2 * 999);
    CSR st = graph::gen::star(100000);
    CHECK(st.n == 100001 && st.degree(0) == 100000 && st.degree(5) == 1);
    CSR k = graph::gen::complete(200);
    CHECK(k.m == 200ull * 199);
    CHECK(graph::gen::empty(1000).m == 0);
    CSR gr = graph::gen::grid2d(100, 100);
    CHECK(gr.m == 2ull * 2 * 100 * 99);
    CSR r = graph::gen::rmat(12, 16, true, 43);
    CHECK(r.directed() && r.n == 4096 && r.m > 30000);
}

TEST(csr_save_load_roundtrip) {
    for (auto& [name, g] : graph::gen::suite()) {
        std::string f = tmpPath(name + ".bin");
        graph::save(g, f);
        CSR h = graph::load(f);
        unlink(f.c_str());
        CHECK(graph::equal(g, h));
        CHECK(graph::validate(h).empty());
    }
}

TEST(load_rejects_bad_files) {
    std::string f = tmpPath("bad.bin");
    FILE* fp = std::fopen(f.c_str(), "wb");
    std::fputs("definitely not a graph file, but longer than a header", fp);
    std::fclose(fp);
    bool threw = false;
    try { graph::load(f); } catch (const std::exception&) { threw = true; }
    unlink(f.c_str());
    CHECK(threw);
}

TEST(transpose_of_transpose_is_identity) {
    for (auto& [name, g] : graph::gen::suite()) {
        CSR t = graph::transpose(g);
        CHECK(graph::validate(t).empty());
        CHECK(graph::equal(graph::transpose(t), g));
        if (!g.directed()) CHECK(graph::equal(t, g));  // symmetric graphs are self-transpose
    }
    // Edge-level check on the directed graph: u->v in G iff v->u in G^T.
    CSR g = graph::gen::rmat(12, 16, true, 43);
    CSR t = graph::transpose(g);
    for (uint32_t u = 0; u < g.n; ++u)
        for (const uint32_t* p = g.begin(u); p != g.end(u); ++p) CHECK(hasEdge(t, *p, u));
}

TEST(symmetrize_is_symmetric) {
    CSR g = graph::gen::rmat(12, 16, true, 43);
    CSR s = graph::symmetrize(g);
    CHECK(!s.directed());
    CHECK(graph::validate(s).empty());
    CHECK(graph::equal(graph::transpose(s), CSR(s)));  // s == s^T
    for (uint32_t u = 0; u < g.n; ++u)                   // contains every original edge both ways
        for (const uint32_t* p = g.begin(u); p != g.end(u); ++p)
            CHECK(hasEdge(s, u, *p) && hasEdge(s, *p, u));
    CHECK(s.m <= 2 * g.m && s.m >= g.m);
    // Undirected input: returned as-is, sharing storage.
    CSR gr = graph::gen::grid2d(10, 10);
    CHECK(graph::symmetrize(gr).col_idx == gr.col_idx);
}

// Converter on a hand-written SNAP-style file covering the awkward cases:
// comments, blank lines, tabs/spaces/CRLF, sparse and 1-based IDs, duplicates,
// self-loops, an ID seen only as a target, and no trailing newline.
TEST(converter_handles_snap_quirks) {
    const char* text =
        "# Directed graph: toy.txt\n"
        "# FromNodeId\tToNodeId\n"
        "10\t20\n"
        "20 30\r\n"
        "\n"
        "10\t20\n"          // duplicate
        "30\t30\n"          // self-loop
        "% another comment style\n"
        "  30\t10  \n"
        "1000\t20\n"
        "20\t5000";         // 5000 only ever a target; no trailing newline
    // Dense IDs in raw-ID order: 10->0, 20->1, 30->2, 1000->3, 5000->4.
    EdgeList dir = {{0, 1}, {1, 2}, {2, 0}, {3, 1}, {1, 4}};

    for (bool gz : {true, false}) {
        std::string in = tmpPath(gz ? "toy.txt.gz" : "toy.txt");
        if (gz) {
            gzFile f = gzopen(in.c_str(), "wb");
            gzputs(f, text);
            gzclose(f);
        } else {
            FILE* f = std::fopen(in.c_str(), "wb");
            std::fputs(text, f);
            std::fclose(f);
        }
        for (bool directed : {true, false}) {
            std::string out = tmpPath("toy.bin");
            graph::ConvertStats st = graph::convertSnap(in, out, directed);
            CSR g = graph::load(out);
            unlink(out.c_str());
            CHECK(st.lines == 7 && st.n == 5 && st.raw_max_id == 5000);
            CHECK(graph::validate(g).empty());
            CHECK(graph::equal(g, graph::fromEdgeList(5, dir, directed)));
            CHECK(g.directed() == directed);
            CHECK(g.m == (directed ? 5u : 10u));
        }
        unlink(in.c_str());
    }
}

TEST(converter_rejects_malformed_input) {
    std::string in = tmpPath("bad.txt");
    FILE* f = std::fopen(in.c_str(), "wb");
    std::fputs("1 2\n3\n", f);  // second line has one field
    std::fclose(f);
    bool threw = false;
    try { graph::convertSnap(in, tmpPath("bad_out.bin"), true); } catch (const std::exception&) { threw = true; }
    unlink(in.c_str());
    CHECK(threw);
}

// Larger converter round-trip: dump an R-MAT graph as text, convert, compare.
TEST(converter_matches_in_memory_build) {
    CSR g = graph::gen::rmat(12, 16, true, 7);
    std::string in = tmpPath("rmat.txt.gz"), out = tmpPath("rmat.bin");
    gzFile f = gzopen(in.c_str(), "wb");
    gzprintf(f, "# rmat dump\n");
    // Shift IDs by 1 (SNAP-style 1-based); relabeling must undo the shift.
    for (uint32_t u = 0; u < g.n; ++u)
        for (const uint32_t* p = g.begin(u); p != g.end(u); ++p) gzprintf(f, "%u\t%u\n", u + 1, *p + 1);
    gzclose(f);
    graph::convertSnap(in, out, true);
    CSR h = graph::load(out);
    unlink(in.c_str());
    unlink(out.c_str());
    // Isolated R-MAT vertices never appear in the text, so compare after
    // dropping them from g.
    std::vector<uint32_t> keep(g.n, cpu::kUnreached);
    std::vector<uint8_t> present(g.n, 0);
    for (uint32_t u = 0; u < g.n; ++u)
        for (const uint32_t* p = g.begin(u); p != g.end(u); ++p) present[u] = present[*p] = 1;
    uint32_t n2 = 0;
    for (uint32_t v = 0; v < g.n; ++v) if (present[v]) keep[v] = n2++;
    EdgeList e;
    for (uint32_t u = 0; u < g.n; ++u)
        for (const uint32_t* p = g.begin(u); p != g.end(u); ++p) e.push_back({keep[u], keep[*p]});
    CHECK(graph::equal(h, graph::fromEdgeList(n2, e, true)));
}

TEST(bfs_serial_known_answers) {
    auto d = cpu::bfsSerial(graph::gen::path(1000), 0);
    for (uint32_t i = 0; i < 1000; ++i) CHECK(d[i] == i);
    d = cpu::bfsSerial(graph::gen::star(100000), 7);
    CHECK(d[7] == 0 && d[0] == 1 && d[8] == 2);
    d = cpu::bfsSerial(graph::gen::twoComponents(), 0);
    CHECK(d[299] == 299 && d[300] == cpu::kUnreached && d[399] == cpu::kUnreached);
    d = cpu::bfsSerial(graph::gen::grid2d(100, 100), 0);
    CHECK(d[99 * 100 + 99] == 198);
    d = cpu::bfsSerial(graph::gen::empty(1000), 3);
    CHECK(d[3] == 0 && d[4] == cpu::kUnreached);
}


// ---------------------------------------------------------------- Phase 2

// Sources for exhaustive checks: first, middle and last vertex plus a few
// random ones; isolated vertices included on purpose (dist = 0 at the source only).
static std::vector<uint32_t> testSources(const CSR& g) {
    std::vector<uint32_t> s = {0, (uint32_t)(g.n / 2), (uint32_t)(g.n - 1)};
    uint64_t x = 12345;
    for (int i = 0; i < 3; ++i) { x = x * 6364136223846793005ull + 1442695040888963407ull; s.push_back((uint32_t)((x >> 33) % g.n)); }
    return s;
}

static uint64_t reachedCount(const std::vector<uint32_t>& d) {
    uint64_t c = 0;
    for (uint32_t x : d) c += x != cpu::kUnreached;
    return c;
}

TEST(cpu_omp_matches_serial_all_schedules_threads) {
    for (auto& [name, g0] : graph::gen::suite()) {
        CSR g = graph::symmetrize(g0);
        for (uint32_t src : testSources(g)) {
            auto ref = cpu::bfsSerial(g, src);
            for (auto sch : {cpu::Schedule::Static, cpu::Schedule::Dynamic64, cpu::Schedule::Guided})
                for (int nt : {1, 3, 8}) {
                    cpu::BfsOptions o;
                    o.schedule = sch;
                    o.threads = nt;
                    std::vector<uint32_t> d(g.n);
                    std::vector<util::BfsLevel> tr;
                    cpu::bfsOmp(g, src, d.data(), o, &tr);
                    if (d != ref) std::printf("    %s src %u sched %s threads %d\n", name.c_str(), src, cpu::scheduleName(sch), nt);
                    CHECK(d == ref);
                    uint64_t sum = 0;
                    for (auto& L : tr) sum += L.frontier;
                    CHECK(sum == reachedCount(ref));
                }
        }
    }
}

TEST(cpu_do_matches_serial_forced_directions) {
    struct AB { double a, b; const char* what; };
    const AB modes[] = {{15, 18, "default"}, {1e-12, 18, "all top-down"},
                        {1e12, 1e12, "all bottom-up"}, {2, 4, "frequent switching"}};
    for (auto& [name, g0] : graph::gen::suite()) {
        CSR g = graph::symmetrize(g0);
        for (uint32_t src : testSources(g)) {
            auto ref = cpu::bfsSerial(g, src);
            for (auto m : modes)
                for (int nt : {1, 8}) {
                    cpu::BfsOptions o;
                    o.alpha = m.a;
                    o.beta = m.b;
                    o.threads = nt;
                    std::vector<uint32_t> d(g.n);
                    std::vector<util::BfsLevel> tr;
                    cpu::bfsDO(g, src, d.data(), o, &tr);
                    if (d != ref) std::printf("    %s src %u mode %s threads %d\n", name.c_str(), src, m.what, nt);
                    CHECK(d == ref);
                    uint64_t sum = 0;
                    bool sawBU = false;
                    for (auto& L : tr) { sum += L.frontier; sawBU |= L.direction == 'B'; }
                    CHECK(sum == reachedCount(ref));
                    if (m.a > 1e6 && reachedCount(ref) > 1) CHECK(sawBU);
                    if (m.a < 1e-6) CHECK(!sawBU);
                }
        }
    }
}

// ---------------------------------------------------------------- Phase 4

// Sizes around every boundary the kernels care about: empty, single element,
// threadgroup (256), scan block (1024), two-level and three-level scans
// (1024^2 = 1048576), and a large odd size.
static const std::vector<size_t> kPrimSizes = {0, 1, 2, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025,
                                               4097, 100003, 1048575, 1048576, 1048577, 3000017};

static std::vector<uint32_t> randomVec(size_t n, uint64_t seed, uint32_t bound) {
    util::Rng r(seed);
    std::vector<uint32_t> v(n);
    for (auto& x : v) x = (uint32_t)r.below(bound);
    return v;
}

static gpu::Primitives& prims() { static gpu::Primitives p(ctx()); return p; }

TEST(primitives_reduce_sum_max) {
    for (size_t n : kPrimSizes) {
        // Full 32-bit range: sums wrap, and both sides wrap identically.
        auto v = randomVec(n, 100 + n, 0xFFFFFFFFu);
        uint32_t sum = 0, mx = 0;
        for (uint32_t x : v) { sum += x; mx = std::max(mx, x); }
        if (prims().runReduceSum(v) != sum || prims().runReduceMax(v) != mx)
            std::printf("    size %zu\n", n);
        CHECK(prims().runReduceSum(v) == sum);
        CHECK(prims().runReduceMax(v) == mx);
    }
}

TEST(primitives_exclusive_scan) {
    for (size_t n : kPrimSizes) {
        auto v = randomVec(n, 200 + n, 1000);
        std::vector<uint32_t> want(n);
        uint32_t run = 0;
        for (size_t i = 0; i < n; ++i) { want[i] = run; run += v[i]; }
        uint32_t total = 0;
        auto got = prims().runScan(v, &total);
        if (got != want || total != run) std::printf("    size %zu\n", n);
        CHECK(got == want);
        CHECK(total == run);
    }
}

TEST(primitives_scan_in_place_and_fill) {
    const size_t n = 1048577;
    auto v = randomVec(n, 7, 50);
    id<MTLBuffer> buf = ctx().buffer(v.data(), n * 4), tot = ctx().buffer(4);
    id<MTLCommandBuffer> cb = [ctx().queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    prims().scanExclusive(enc, buf, buf, n, tot);  // in place
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const uint32_t* o = static_cast<const uint32_t*>(buf.contents);
    uint32_t run = 0;
    for (size_t i = 0; i < n; ++i) { CHECK(o[i] == run); run += v[i]; }
    CHECK(*static_cast<uint32_t*>(tot.contents) == run);

    for (size_t m : {(size_t)0, (size_t)1, (size_t)257, (size_t)100003}) {
        id<MTLBuffer> b = ctx().buffer((m + 2) * 4);
        uint32_t* p = static_cast<uint32_t*>(b.contents);
        for (size_t i = 0; i < m + 2; ++i) p[i] = 1;
        cb = [ctx().queue() commandBuffer];
        enc = [cb computeCommandEncoder];
        prims().fill(enc, b, 0xABCD1234u, m, 4);  // offset 1 element: p[0] and p[m+1] untouched
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        CHECK(p[0] == 1 && p[m + 1] == 1);
        for (size_t i = 1; i <= m; ++i) CHECK(p[i] == 0xABCD1234u);
    }
}

TEST(primitives_queue_bitmap_roundtrip) {
    for (size_t n : {(size_t)0, (size_t)1, (size_t)31, (size_t)32, (size_t)33, (size_t)1000, (size_t)100003}) {
        // Random subset of 0..n-1 (about a third), as an unordered queue.
        util::Rng r(300 + n);
        std::vector<uint32_t> queue;
        for (uint32_t v = 0; v < n; ++v) if (r.below(3) == 0) queue.push_back(v);
        for (size_t i = queue.size(); i > 1; --i) std::swap(queue[i - 1], queue[r.below(i)]);

        size_t words = (n + 31) / 32;
        id<MTLBuffer> q = ctx().buffer(queue.data(), queue.size() * 4);
        id<MTLBuffer> bm = ctx().buffer(words * 4), q2 = ctx().buffer(n * 4), sz = ctx().buffer(4);
        std::memset(bm.contents, 0xFF, words * 4);  // must be cleared by queueToBitmap
        id<MTLCommandBuffer> cb = [ctx().queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        prims().queueToBitmap(enc, q, queue.size(), bm, n);
        prims().bitmapToQueue(enc, bm, n, q2, sz);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        const uint32_t* bits = static_cast<const uint32_t*>(bm.contents);
        std::vector<uint8_t> in(n, 0);
        for (uint32_t v : queue) in[v] = 1;
        for (uint32_t v = 0; v < n; ++v) CHECK(((bits[v >> 5] >> (v & 31)) & 1u) == in[v]);
        if (n % 32) CHECK((bits[words - 1] >> (n % 32)) == 0);  // no bits past n

        uint32_t count = *static_cast<uint32_t*>(sz.contents);
        CHECK(count == queue.size());
        const uint32_t* o = static_cast<const uint32_t*>(q2.contents);
        std::vector<uint32_t> got(o, o + count), want = queue;
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        CHECK(got == want);
    }
}

// Every GPU BFS variant (and gpu_binned / gpu_edge parameter extremes),
// every synthetic graph, several sources (including
// isolated vertices), plus the vertex with the highest degree (star hub):
// dist must equal cpu_serial exactly, and the trace must account for every
// reached vertex exactly once.
TEST(gpu_bfs_matches_serial_on_synthetic_graphs) {
    for (auto& [name, stored] : graph::gen::suite()) {
        CSR g = graph::symmetrize(stored);
        gpu::GpuBfs bfs(ctx(), g);
        auto sources = testSources(g);
        uint32_t hub = 0;
        for (uint32_t v = 0; v < g.n; ++v) if (g.degree(v) > g.degree(hub)) hub = v;
        sources.push_back(hub);
        for (uint32_t src : sources) {
            auto ref = cpu::bfsSerial(g, src);
            // gpu_binned threshold sets: default, everything small, everything
            // large, everything medium-or-large, tiny bins. gpu_edge: several
            // edges-per-thread values (1 = one edge per thread, 16 = long chunks).
            struct Cfg { const char* name; int variant; uint32_t smallMax, largeMin, ept; };
            const Cfg cfgs[] = {
                {"gpu_vertex", 0, 0, 0, 0},           {"gpu_naive", 1, 0, 0, 0},
                {"gpu_binned", 2, 32, 1024, 0},       {"gpu_binned/all-small", 2, 1u << 30, 1u << 31, 0},
                {"gpu_binned/all-large", 2, 0, 0, 0}, {"gpu_binned/medium", 2, 1, 1u << 30, 0},
                {"gpu_binned/2-3", 2, 2, 3, 0},       {"gpu_edge/1", 3, 0, 0, 1},
                {"gpu_edge/4", 3, 0, 0, 4},           {"gpu_edge/16", 3, 0, 0, 16},
            };
            for (const Cfg& c : cfgs)
                for (uint32_t tg : {32u, 256u, 1024u}) {
                    gpu::GpuBfsOptions o;
                    o.tg = tg;
                    if (c.variant == 2) { o.smallMax = c.smallMax; o.largeMin = c.largeMin; }
                    if (c.variant == 3) o.ept = c.ept;
                    std::vector<util::BfsLevel> tr;
                    std::vector<uint32_t> d(g.n, 12345);
                    if (c.variant == 0) bfs.runVertex(src, d.data(), o, &tr);
                    else if (c.variant == 1) bfs.runNaive(src, d.data(), o, &tr);
                    else if (c.variant == 2) bfs.runBinned(src, d.data(), o, &tr);
                    else bfs.runEdge(src, d.data(), o, &tr);
                    if (d != ref) std::printf("    %s src %u %s tg %u\n", name.c_str(), src, c.name, tg);
                    CHECK(d == ref);
                    uint64_t sum = 0, edges = 0, want = 0;
                    for (auto& L : tr) { sum += L.frontier; edges += L.edges; }
                    for (uint32_t v = 0; v < g.n; ++v) if (ref[v] != cpu::kUnreached) want += g.degree(v);
                    CHECK(sum == reachedCount(ref));
                    CHECK(edges == want);  // top-down examines every reached vertex's list once
                }
        }
        // The CSR view aliases the GPU buffers and matches the input exactly.
        CHECK(graph::equal(bfs.graph(), g));
        // Zero-copy path: out == dist().
        bfs.runNaive(sources[0], bfs.dist(), gpu::GpuBfsOptions{});
        auto ref = cpu::bfsSerial(g, sources[0]);
        CHECK(std::memcmp(bfs.dist(), ref.data(), g.n * 4) == 0);
    }
}

// gpu_do: exact dist; identical per-level direction decisions to cpu_do
// (same GAP rule on the same counts); forced modes (alpha huge -> bottom-up
// as early as possible, alpha tiny -> never bottom-up) exercise both
// conversions and both kernels on every graph.
TEST(gpu_do_matches_serial_and_cpu_decisions) {
    struct Mode { const char* what; double a, b; };
    const Mode modes[] = {{"default", 15, 18}, {"force-bu", 1e9, 1e9}, {"never-bu", 1e-9, 18},
                          {"flip-flop", 2, 2}};
    for (auto& [name, stored] : graph::gen::suite()) {
        CSR g = graph::symmetrize(stored);
        gpu::GpuBfs bfs(ctx(), g);
        auto sources = testSources(g);
        uint32_t hub = 0;
        for (uint32_t v = 0; v < g.n; ++v) if (g.degree(v) > g.degree(hub)) hub = v;
        sources.push_back(hub);
        for (uint32_t src : sources) {
            auto ref = cpu::bfsSerial(g, src);
            for (const Mode& m : modes)
                for (uint32_t tg : {32u, 256u}) {
                    gpu::GpuBfsOptions o;
                    o.tg = tg;
                    o.alpha = m.a;
                    o.beta = m.b;
                    std::vector<util::BfsLevel> tr, ctr;
                    std::vector<uint32_t> d(g.n, 777);
                    bfs.runDO(src, d.data(), o, &tr);
                    if (d != ref) std::printf("    %s src %u mode %s tg %u\n", name.c_str(), src, m.what, tg);
                    CHECK(d == ref);
                    uint64_t sum = 0;
                    bool sawBU = false;
                    for (auto& L : tr) { sum += L.frontier; sawBU |= L.direction == 'B'; }
                    CHECK(sum == reachedCount(ref));
                    if (m.a > 1e6 && reachedCount(ref) > 1) CHECK(sawBU);
                    if (m.a < 1e-6) CHECK(!sawBU);

                    cpu::BfsOptions co;
                    co.alpha = m.a;
                    co.beta = m.b;
                    std::vector<uint32_t> cd(g.n);
                    cpu::bfsDO(g, src, cd.data(), co, &ctr);
                    CHECK(tr.size() == ctr.size());
                    for (size_t i = 0; i < tr.size(); ++i) {
                        CHECK(tr[i].direction == ctr[i].direction);
                        CHECK(tr[i].frontier == ctr[i].frontier);
                    }
                }
        }
    }
}

// hybrid: every threshold from all-GPU to all-CPU must give exact dist and
// the same direction sequence as cpu_do (the device choice must not change
// the algorithm); with T = inf no level may touch the GPU, with T = 0 none
// the CPU, and intermediate T must put small levels on the CPU.
TEST(hybrid_matches_serial_at_every_threshold) {
    const uint64_t Ts[] = {0, 1, 64, 1000, 100000, UINT64_MAX};
    struct Mode { double a, b; };
    const Mode modes[] = {{15, 18}, {1e9, 1e9}, {2, 2}};
    for (auto& [name, stored] : graph::gen::suite()) {
        CSR g = graph::symmetrize(stored);
        gpu::GpuBfs bfs(ctx(), g);
        auto sources = testSources(g);
        uint32_t hub = 0;
        for (uint32_t v = 0; v < g.n; ++v) if (g.degree(v) > g.degree(hub)) hub = v;
        sources.push_back(hub);
        for (uint32_t src : sources) {
            auto ref = cpu::bfsSerial(g, src);
            for (const Mode& m : modes) {
                cpu::BfsOptions co;
                co.alpha = m.a;
                co.beta = m.b;
                std::vector<util::BfsLevel> ctr;
                std::vector<uint32_t> cd(g.n);
                cpu::bfsDO(g, src, cd.data(), co, &ctr);
                for (uint64_t T : Ts)
                    for (int threads : {1, 8}) {
                        gpu::GpuBfsOptions o;
                        o.alpha = m.a;
                        o.beta = m.b;
                        o.hybridT = T;
                        o.cpuThreads = threads;
                        std::vector<util::BfsLevel> tr;
                        std::vector<uint32_t> d(g.n, 777);
                        bfs.runHybrid(src, d.data(), o, &tr);
                        if (d != ref)
                            std::printf("    %s src %u T %llu threads %d\n", name.c_str(), src,
                                        (unsigned long long)T, threads);
                        CHECK(d == ref);
                        CHECK(tr.size() == ctr.size());
                        bool cpuLevel = false, gpuLevel = false;
                        for (size_t i = 0; i < tr.size(); ++i) {
                            CHECK(tr[i].direction == ctr[i].direction);
                            CHECK(tr[i].frontier == ctr[i].frontier);
                            cpuLevel |= tr[i].device == 'C';
                            gpuLevel |= tr[i].device == 'G';
                        }
                        if (T == 0) CHECK(!cpuLevel);
                        if (T == UINT64_MAX) CHECK(!gpuLevel);
                        // Level 0 top-down examines degree(source) edges: CPU iff below T.
                        if (tr[0].direction == 'T')
                            CHECK((tr[0].device == 'C') == (g.degree(src) < T));
                    }
            }
        }
    }
}

// ---------------------------------------------------------------- Phase 8

// Hand-checked fixed points. 2 vertices, edge 0->1, vertex 1 dangling:
// r0 = 0.075 + 0.425 r1, r0 + r1 = 1  =>  r0 = 0.5 / 1.425. Cycles,
// complete graphs and the empty graph (all dangling) are uniform.
TEST(pagerank_cpu_known_answers) {
    cpu::PrOptions o;
    o.tol = 1e-14;
    o.iters = 1000;
    auto solve = [&](const CSR& g, bool parallel) {
        CSR in = graph::transpose(g);
        auto od = cpu::outDegrees(g);
        std::vector<double> r(g.n);
        if (parallel) cpu::pagerankOmp(in, od.data(), r.data(), o);
        else cpu::pagerankSerial(in, od.data(), r.data(), o);
        return r;
    };
    for (bool par : {false, true}) {
        auto r = solve(graph::fromEdgeList(2, {{0, 1}}, true), par);
        CHECK(std::fabs(r[0] - 0.5 / 1.425) < 1e-12 && std::fabs(r[1] - (1 - 0.5 / 1.425)) < 1e-12);
        r = solve(graph::fromEdgeList(3, {{0, 1}, {1, 2}, {2, 0}}, true), par);
        for (double x : r) CHECK(std::fabs(x - 1.0 / 3) < 1e-12);
        r = solve(graph::gen::complete(200), par);
        for (double x : r) CHECK(std::fabs(x - 1.0 / 200) < 1e-12);
        r = solve(graph::gen::empty(1000), par);
        for (double x : r) CHECK(std::fabs(x - 1.0 / 1000) < 1e-12);
    }
}

// Every GPU variant (and cpu_omp) vs cpu_serial on every synthetic graph:
// fixed 20 iterations and convergence to 1e-6. The converged iteration count
// must match the reference within 3: near convergence the per-vertex change
// is ~1e-9 on path1000, the difference of two floats near 1e-3 (ulp 1.2e-10),
// so the float L1 delta has ~10% noise around the tolerance; with the delta
// shrinking only x0.85 per iteration that moves the stop by 1-2 iterations
// (observed: 49 vs 47). The ranks themselves must still pass checkRanks.
// Runs that stop at the float precision floor (GpuPrResult::floorStop; the
// star's hub makes the float L1 change stagnate near 1e-6) skip the
// iteration-count check; gpu_push's float-atomic order also makes its count
// vary on the star (69-77 vs 90), so it is held to the rank check only there.
TEST(pagerank_variants_match_reference) {
    using gpu::PrVariant;
    struct Cfg { const char* name; PrVariant v; uint32_t smallMax, largeMin; };
    const Cfg cfgs[] = {
        {"push", PrVariant::Push, 16, 256},          {"pull_scalar", PrVariant::PullScalar, 16, 256},
        {"pull_vector", PrVariant::PullVector, 16, 256}, {"adaptive", PrVariant::PullAdaptive, 16, 256},
        {"adaptive/all-small", PrVariant::PullAdaptive, 1u << 30, 1u << 31},
        {"adaptive/all-large", PrVariant::PullAdaptive, 0, 0},
        {"adaptive/all-medium", PrVariant::PullAdaptive, 0, 1u << 30},
        {"adaptive/2-3", PrVariant::PullAdaptive, 2, 3},
    };
    for (auto& [name, g] : graph::gen::suite()) {
        CSR in = g.directed() ? graph::transpose(g) : g;
        auto od = cpu::outDegrees(g);
        gpu::GpuPageRank gpr(ctx(), g, in, od, true, true);
        for (double tol : {0.0, 1e-6}) {
            cpu::PrOptions co;
            co.tol = tol;
            co.iters = tol > 0 ? 1000 : 20;
            std::vector<double> ref(g.n), omp(g.n);
            int refIters = cpu::pagerankSerial(in, od.data(), ref.data(), co).iters;
            cpu::pagerankOmp(in, od.data(), omp.data(), co);
            CHECK(cpu::checkRanks(ref, omp.data()).l1 < 1e-12);
            for (const Cfg& c : cfgs)
                for (uint32_t tg : {32u, 256u, 1024u}) {
                    gpu::GpuPrOptions o;
                    o.tol = tol;
                    o.iters = co.iters;
                    o.tg = tg;
                    o.smallMax = c.smallMax;
                    o.largeMin = c.largeMin;
                    gpu::GpuPrResult r = gpr.run(c.v, o);
                    cpu::PrCheck chk = cpu::checkRanks(ref, gpr.ranks());
                    if (!chk.ok())
                        std::printf("    %s %s tol %g tg %u: L1 %.3g rel %.3g sum %.8f\n", name.c_str(), c.name,
                                    tol, tg, chk.l1, chk.maxRelTop, chk.sum);
                    CHECK(chk.ok());
                    if (r.floorStop) continue;  // float floor reached (star100k): ranks checked above
                    if (c.v == PrVariant::Push && name == "star100k") continue;
                    if (std::abs(r.iters - refIters) > 3)
                        std::printf("    %s %s tg %u: iters %d vs reference %d (last delta %.3g)\n", name.c_str(),
                                    c.name, tg, r.iters, refIters, r.lastDelta);
                    CHECK(std::abs(r.iters - refIters) <= 3);
                }
        }
    }
}

// ---------------------------------------------------------------- Phase 9

// Every ordering is a permutation; permute() then the inverse permutation
// restores the graph exactly; BFS distances and PageRank ranks of the
// relabeled graph map back (v -> perm[v]) to the original results.
TEST(reorder_valid_and_results_map_back) {
    for (auto& [name, stored] : graph::gen::suite()) {
        CSR sym = graph::symmetrize(stored);
        for (const char* ord : {"original", "random", "degree", "rcm"}) {
            auto perm = graph::reorder::compute(sym, ord);
            CHECK(graph::reorder::isPermutation(perm));
            CSR p = graph::permute(stored, perm);
            CHECK(graph::validate(p).empty());
            std::vector<uint32_t> inv(perm.size());
            for (size_t v = 0; v < perm.size(); ++v) inv[perm[v]] = (uint32_t)v;
            CHECK(graph::equal(graph::permute(p, inv), stored));

            CSR psym = graph::permute(sym, perm);
            for (uint32_t src : testSources(sym)) {
                auto ref = cpu::bfsSerial(sym, src);
                auto got = cpu::bfsSerial(psym, perm[src]);
                for (uint32_t v = 0; v < sym.n; ++v) CHECK(got[perm[v]] == ref[v]);
            }
            cpu::PrOptions o;
            CSR in = stored.directed() ? graph::transpose(stored) : stored;
            CSR pin = p.directed() ? graph::transpose(p) : p;
            auto od = cpu::outDegrees(stored), pod = cpu::outDegrees(p);
            std::vector<double> r(stored.n), pr(stored.n);
            cpu::pagerankSerial(in, od.data(), r.data(), o);
            cpu::pagerankSerial(pin, pod.data(), pr.data(), o);
            // Summation order differs after relabeling: equal to rounding.
            for (uint32_t v = 0; v < stored.n; ++v) CHECK(std::fabs(pr[perm[v]] - r[v]) <= 1e-12 * r[v] + 1e-18);
        }
    }
}

// Known locality answers: RCM recovers a path's natural order from a random
// labeling (mean gap 1, i.e. every edge joins consecutive ids); degree order
// puts the star's hub first; random labels destroy a grid's locality.
TEST(reorder_locality_known_answers) {
    CSR path = graph::gen::path(1000);
    CSR shuffled = graph::permute(path, graph::reorder::compute(path, "random"));
    CHECK(graph::reorder::locality(shuffled).meanGap > 100);
    CSR rcm = graph::permute(shuffled, graph::reorder::compute(shuffled, "rcm"));
    CHECK(std::fabs(graph::reorder::locality(rcm).meanGap - 1.0) < 1e-12);

    CSR star = graph::gen::star(1000);
    auto dp = graph::reorder::compute(star, "degree");
    uint32_t hub = 0;
    for (uint32_t v = 0; v < star.n; ++v) if (star.degree(v) > star.degree(hub)) hub = v;
    CHECK(dp[hub] == 0);

    CSR grid = graph::gen::grid2d(100, 100);
    auto lg = graph::reorder::locality(grid);
    auto lr = graph::reorder::locality(graph::permute(grid, graph::reorder::compute(grid, "random")));
    CHECK(lr.meanGap > 10 * lg.meanGap);
    CHECK(lr.sameBlock64 < lg.sameBlock64);
}

// ---------------------------------------------------------------- runner

// gtest [substring]: run only tests whose name contains the substring.
int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    int passed = 0;
    for (auto& t : registry()) {
        if (!filter.empty() && t.name.find(filter) == std::string::npos) continue;
        g_current_failed = false;
        std::printf("[ RUN  ] %s\n", t.name.c_str());
        auto t0 = std::chrono::steady_clock::now();
        try {
            t.fn();
        } catch (const std::exception& e) {
            std::printf("    unexpected exception: %s\n", e.what());
            g_current_failed = true;
        }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (g_current_failed) { ++g_failures; std::printf("[ FAIL ] %s (%.1f s)\n", t.name.c_str(), secs); }
        else { ++passed; std::printf("[  OK  ] %s (%.1f s)\n", t.name.c_str(), secs); }
    }
    std::printf("\n%d passed, %d failed\n", passed, g_failures);
    return g_failures ? 1 : 0;
}
