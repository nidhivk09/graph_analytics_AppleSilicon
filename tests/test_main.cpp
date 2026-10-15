// gtest: tiny assert-based test runner (no googletest dependency).
// Register tests with TEST(name) { ... } and check with CHECK(cond).
// A failing CHECK records the failure and continues with the next test.
#include "cpu/bfs_cpu.hpp"
#include "graph/csr.hpp"
#include "graph/generators.hpp"
#include "graph/snap.hpp"
#include "graph/transform.hpp"
#include "metal/bfs_gpu.hpp"
#include "metal/context.hpp"
#include "metal/primitives.hpp"
#include "util/rng.hpp"

#include <cstdlib>
#include <unistd.h>
#include <zlib.h>

#include <omp.h>
#include <cstdio>
#include <algorithm>
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

// Every GPU BFS variant, every synthetic graph, several sources (including
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
            for (int variant = 0; variant < 2; ++variant)
                for (uint32_t tg : {32u, 256u, 1024u}) {
                    gpu::GpuBfsOptions o;
                    o.tg = tg;
                    std::vector<util::BfsLevel> tr;
                    std::vector<uint32_t> d(g.n, 12345);
                    if (variant == 0) bfs.runVertex(src, d.data(), o, &tr);
                    else bfs.runNaive(src, d.data(), o, &tr);
                    if (d != ref)
                        std::printf("    %s src %u %s tg %u\n", name.c_str(), src,
                                    variant ? "gpu_naive" : "gpu_vertex", tg);
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

// ---------------------------------------------------------------- runner

int main() {
    int passed = 0;
    for (auto& t : registry()) {
        g_current_failed = false;
        std::printf("[ RUN  ] %s\n", t.name.c_str());
        try {
            t.fn();
        } catch (const std::exception& e) {
            std::printf("    unexpected exception: %s\n", e.what());
            g_current_failed = true;
        }
        if (g_current_failed) { ++g_failures; std::printf("[ FAIL ] %s\n", t.name.c_str()); }
        else { ++passed; std::printf("[  OK  ] %s\n", t.name.c_str()); }
    }
    std::printf("\n%d passed, %d failed\n", passed, g_failures);
    return g_failures ? 1 : 0;
}
