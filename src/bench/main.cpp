// gbench: benchmark + validation CLI for every BFS (and later PageRank) variant.
//
//   gbench --graph data/soc-LiveJournal1.bin --algo bfs --variant cpu_omp
//          [--threads 8] [--schedule static|dynamic,64|guided] [--alpha 15 --beta 18]
//          [--tg 256]                                (GPU variants: threadgroup size)
//          [--small-max 16 --large-min 256]         (gpu_binned degree thresholds)
//          [--ept 2]                                 (gpu_edge: edges per thread)
//          [--T 100000 | --T inf]                    (hybrid: levels with work < T run on
//                                                     the CPU with --threads threads, default 4)
//          [--sources 64 | --source S] [--seed 1] [--any-component] [--warmup 2] [--trials 5]
//          [--source-warmup 1]
//          [--validate] [--trace trace.csv] [--csv results/bfs.csv] [--ordering original]
//   gbench --graph data/soc-LiveJournal1.bin --algo pr --variant gpu_pull_vector
//          [--iters 20 | --tol 1e-6 [--iters 1000]] [--tg 256] [--small-max 16 --large-min 256]
//          [--threads 8] [--warmup 2] [--trials 5] [--validate] [--csv results/pr.csv]
//          [--dump-ranks out.f64]
//   gbench --graph synth:star100k ...         use a generated test graph
//   gbench --graph X --save-graph out.bin     write the (stored, unsymmetrized) graph
//   gbench --graph X --variant V --source S --dump-dist out.u32   raw uint32 distances
//   --ordering original|random|degree|rcm   relabel vertices first (untimed; Phase 9),
//          [--locality-csv results/reorder_locality.csv]   and record the locality metrics.
//          Sources, references and TEPS denominators stay in ORIGINAL ids: results on
//          the relabeled graph are mapped back (v -> perm[v]) before validation.
//
// Methodology (PLAN.md 8): BFS runs on the symmetrized graph; sources are
// random vertices with degree > 0 in the largest connected component, from a
// fixed seed (same for every variant);
// TEPS denominator = undirected edges in the source's component, computed
// from the cpu_serial result; 2 warm-up runs, then per source: compute the
// reference, 1 untimed warm-up run (--source-warmup), `trials` timed runs.
// The per-source warm-up exists because the reference BFS (serial, up to
// ~0.5 s) leaves the GPU idle and it clocks down: without it the first timed
// trial of every source ran ~15% slower (Phase 4). Summary: harmonic mean of per-source GTEPS (from the median time)
// and the median / min time over all runs.
//
// GPU variants: the graph is uploaded once (untimed) into shared Metal
// buffers and every later step, including the CPU reference, uses a CSR view
// of those buffers. dist is the GPU's shared output buffer (zero copy). The
// timed region (time_ms_e2e) covers initialization, every level and every
// per-level host synchronization; time_ms_gpu is the kernel-only part.
//
// Compiled as Objective-C++ (Makefile) because GPU variants use Metal types.
#include "cpu/bfs_cpu.hpp"
#include "cpu/pagerank_cpu.hpp"
#include "graph/csr.hpp"
#include "graph/generators.hpp"
#include "graph/reorder.hpp"
#include "graph/transform.hpp"
#include "metal/bfs_gpu.hpp"
#include "metal/pagerank_gpu.hpp"
#include "util/args.hpp"
#include "util/csv.hpp"
#include "util/rng.hpp"
#include "util/sysinfo.hpp"
#include "util/timer.hpp"

#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <sys/stat.h>

using graph::CSR;

namespace {

using BfsFn = std::function<void(const CSR&, uint32_t, uint32_t*, const cpu::BfsOptions&,
                                 std::vector<util::BfsLevel>*)>;

const std::vector<std::string> kVariants = {"cpu_serial", "cpu_omp",   "cpu_do",  "gpu_vertex",
                                            "gpu_naive",  "gpu_binned", "gpu_edge", "gpu_do",
                                            "hybrid"};

// Variants that need the graph uploaded to Metal buffers.
bool isGpuVariant(const std::string& v) { return v.rfind("gpu_", 0) == 0 || v == "hybrid"; }

// The function that runs `variant`. GPU variants are bound to `gbfs` (the
// uploaded graph) and ignore the CSR argument.
BfsFn bfsVariant(const std::string& variant, gpu::GpuBfs* gbfs, const gpu::GpuBfsOptions& gopt) {
    if (variant == "cpu_serial")
        return [](const CSR& g, uint32_t s, uint32_t* d, const cpu::BfsOptions&,
                  std::vector<util::BfsLevel>*) { cpu::bfsSerial(g, s, d); };
    if (variant == "cpu_omp") return cpu::bfsOmp;
    if (variant == "cpu_do") return cpu::bfsDO;
    if (variant == "gpu_vertex")
        return [gbfs, gopt](const CSR&, uint32_t s, uint32_t* d, const cpu::BfsOptions&,
                            std::vector<util::BfsLevel>* tr) { gbfs->runVertex(s, d, gopt, tr); };
    if (variant == "gpu_naive")
        return [gbfs, gopt](const CSR&, uint32_t s, uint32_t* d, const cpu::BfsOptions&,
                            std::vector<util::BfsLevel>* tr) { gbfs->runNaive(s, d, gopt, tr); };
    if (variant == "gpu_binned")
        return [gbfs, gopt](const CSR&, uint32_t s, uint32_t* d, const cpu::BfsOptions&,
                            std::vector<util::BfsLevel>* tr) { gbfs->runBinned(s, d, gopt, tr); };
    if (variant == "gpu_edge")
        return [gbfs, gopt](const CSR&, uint32_t s, uint32_t* d, const cpu::BfsOptions&,
                            std::vector<util::BfsLevel>* tr) { gbfs->runEdge(s, d, gopt, tr); };
    if (variant == "gpu_do")
        return [gbfs, gopt](const CSR&, uint32_t s, uint32_t* d, const cpu::BfsOptions&,
                            std::vector<util::BfsLevel>* tr) { gbfs->runDO(s, d, gopt, tr); };
    if (variant == "hybrid")
        return [gbfs, gopt](const CSR&, uint32_t s, uint32_t* d, const cpu::BfsOptions&,
                            std::vector<util::BfsLevel>* tr) { gbfs->runHybrid(s, d, gopt, tr); };
    return nullptr;
}

CSR loadGraph(const std::string& spec, std::string& name) {
    if (spec.rfind("synth:", 0) == 0) {
        name = spec.substr(6);
        for (auto& [n, g] : graph::gen::suite())
            if (n == name) return g;
        throw std::runtime_error("unknown synthetic graph '" + name + "'");
    }
    size_t s = spec.find_last_of('/');
    name = spec.substr(s == std::string::npos ? 0 : s + 1);
    name = name.substr(0, name.find('.'));
    CSR g = graph::load(spec);
    return g;
}

// Largest connected component of a symmetric graph, as a membership mask.
std::vector<uint8_t> largestComponent(const CSR& g) {
    std::vector<uint32_t> comp(g.n, cpu::kUnreached), queue(g.n);
    uint32_t best = 0, id = 0;
    uint64_t bestSize = 0;
    for (uint64_t r = 0; r < g.n; ++r, ++id) {
        if (comp[r] != cpu::kUnreached) { --id; continue; }
        size_t head = 0, tail = 0;
        comp[r] = id;
        queue[tail++] = (uint32_t)r;
        while (head < tail) {
            uint32_t u = queue[head++];
            for (const uint32_t* p = g.begin(u); p != g.end(u); ++p)
                if (comp[*p] == cpu::kUnreached) { comp[*p] = id; queue[tail++] = *p; }
        }
        if (tail > bestSize) { bestSize = tail; best = id; }
    }
    std::vector<uint8_t> mask(g.n);
    for (uint64_t v = 0; v < g.n; ++v) mask[v] = comp[v] == best;
    return mask;
}

// Graph500-style sources: distinct random vertices with degree > 0, drawn from
// the largest connected component. Without that restriction, a source in a
// 2-vertex component (roadNet-CA has 2,637 small components) gives a TEPS
// value near zero that dominates the harmonic mean. --any-component disables it.
std::vector<uint32_t> pickSources(const CSR& g, int count, uint64_t seed, bool anyComponent) {
    std::vector<uint8_t> ok(g.n, 1);
    if (!anyComponent) ok = largestComponent(g);
    uint64_t candidates = 0;
    for (uint64_t v = 0; v < g.n; ++v) candidates += (ok[v] && g.degree(v) > 0);
    count = (int)std::min<uint64_t>(count, candidates);
    std::vector<uint32_t> out;
    util::Rng rng(seed);
    std::set<uint32_t> used;
    while ((int)out.size() < count) {
        uint32_t v = (uint32_t)rng.below(g.n);
        if (ok[v] && g.degree(v) > 0 && used.insert(v).second) out.push_back(v);
    }
    return out;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t k = v.size();
    return k == 0 ? 0 : (k % 2 ? v[k / 2] : 0.5 * (v[k / 2 - 1] + v[k / 2]));
}

bool fileExists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }

// --ordering: perm[old] = new, computed on the symmetrized graph (empty for
// "original"). --locality-csv: metrics of the relabeled symmetric graph,
// one row per dataset/ordering (replaced on re-run).
std::vector<uint32_t> orderingPerm(const util::Args& args, const CSR& sym, const std::string& dataset) {
    const std::string ordering = args.get("ordering", "original");
    std::vector<uint32_t> perm;
    util::Timer t;
    if (ordering != "original") perm = graph::reorder::compute(sym, ordering, (uint64_t)args.getInt("order-seed", 1));
    const double orderMs = t.ms();
    if (args.has("locality-csv")) {
        CSR relabeled = perm.empty() ? sym : graph::permute(sym, perm);
        graph::reorder::Locality l = graph::reorder::locality(relabeled);
        char row[256];
        std::snprintf(row, sizeof row, "%s/%s,%s,%s,%llu,%llu,%.2f,%.6f,%.0f", dataset.c_str(), ordering.c_str(),
                      dataset.c_str(), ordering.c_str(), (unsigned long long)sym.n, (unsigned long long)sym.m,
                      l.meanGap, l.sameBlock64, orderMs);
        util::csvUpsert(args.get("locality-csv"), "key,dataset,ordering,n,m_sym,mean_gap,same_block64,order_ms",
                        dataset + "/" + ordering, {row});
        std::printf("locality (%s): mean |u-v| %.1f, same 64-block %.2f%%, ordering took %.0f ms\n",
                    ordering.c_str(), l.meanGap, 100 * l.sameBlock64, orderMs);
    }
    return perm;
}

int runBfs(const util::Args& args, const CSR& stored, const std::string& dataset) {
    const std::string variant = args.get("variant", "cpu_serial");
    if (std::find(kVariants.begin(), kVariants.end(), variant) == kVariants.end()) {
        std::fprintf(stderr, "unknown BFS variant '%s'. Available:", variant.c_str());
        for (auto& k : kVariants) std::fprintf(stderr, " %s", k.c_str());
        std::fprintf(stderr, "\n");
        return 2;
    }
    const bool gpuVariant = isGpuVariant(variant);

    cpu::BfsOptions opt;
    opt.threads = (int)args.getInt("threads", 0);
    opt.schedule = cpu::parseSchedule(args.get("schedule", "dynamic,64"));
    opt.alpha = args.getDouble("alpha", 15.0);
    opt.beta = args.getDouble("beta", 18.0);
    gpu::GpuBfsOptions gopt;
    gopt.tg = (uint32_t)args.getInt("tg", 256);
    gopt.smallMax = (uint32_t)args.getInt("small-max", 16);
    gopt.largeMin = (uint32_t)args.getInt("large-min", 256);
    gopt.ept = (uint32_t)args.getInt("ept", 2);
    gopt.alpha = opt.alpha;  // --alpha / --beta are shared with cpu_do
    gopt.beta = opt.beta;
    // Hybrid CPU levels are small by construction: 4 threads (the P-cores)
    // beat 8 in the Phase 7 sweep, as small levels did in Phase 2.
    gopt.cpuThreads = (variant == "hybrid" && !args.has("threads")) ? 4 : opt.threads;
    const std::string tArg = args.get("T", "100000");
    gopt.hybridT = tArg == "inf" ? UINT64_MAX : (uint64_t)std::stoull(tArg);

    // Transform cost (and the GPU upload) is outside every timed region (PLAN 4.4).
    util::Timer prep;
    CSR g = graph::symmetrize(stored);
    // refG keeps ORIGINAL labels: sources, reference BFS, TEPS denominators.
    // g is what the variant runs on (relabeled if --ordering is not original).
    CSR refG = g;
    const std::vector<uint32_t> perm = orderingPerm(args, g, dataset);
    const bool permuted = !perm.empty();
    if (permuted) g = graph::permute(g, perm);
    auto toRun = [&](uint32_t s) { return permuted ? perm[s] : s; };  // original id -> run id
    std::unique_ptr<gpu::MetalContext> ctx;
    std::unique_ptr<gpu::GpuBfs> gbfs;
    if (gpuVariant) {
        ctx = std::make_unique<gpu::MetalContext>();
        gbfs = std::make_unique<gpu::GpuBfs>(*ctx, g);
        g = gbfs->graph();  // releases the host copy (unless shared with `stored`)
        if (!permuted) refG = g;  // same labels: let the reference use the GPU's bytes too
    }
    double prepMs = prep.ms();
    const BfsFn fn = bfsVariant(variant, gbfs.get(), gopt);

    std::vector<uint32_t> sources;
    if (args.has("source")) sources = {(uint32_t)args.getInt("source", 0)};
    else sources = pickSources(refG, (int)args.getInt("sources", 64), (uint64_t)args.getInt("seed", 1),
                                args.has("any-component"));
    if (sources.empty()) { std::fprintf(stderr, "no vertex with degree > 0\n"); return 1; }
    for (uint32_t s : sources)
        if (s >= g.n) { std::fprintf(stderr, "source %u out of range\n", s); return 2; }

    const int warmup = (int)args.getInt("warmup", 2), trials = (int)args.getInt("trials", 5);
    const int sourceWarmup = (int)args.getInt("source-warmup", 1);
    const bool validate = args.has("validate");
    const std::string ordering = args.get("ordering", "original");
    // dist (run ids) equals ref (original ids) after mapping v -> perm[v].
    auto matches = [&](const uint32_t* d, const std::vector<uint32_t>& r, uint64_t* bad) {
        if (!permuted) {
            if (std::memcmp(d, r.data(), g.n * sizeof(uint32_t)) == 0) return true;
            for (uint64_t v = 0; v < g.n; ++v) if (d[v] != r[v]) { *bad = v; break; }
            return false;
        }
        for (uint64_t v = 0; v < g.n; ++v) if (d[perm[v]] != r[v]) { *bad = v; return false; }
        return true;
    };

    // Only parameters that affect the variant go into the CSV "params" field
    // (space-separated; never commas).
    std::string params;
    if (variant == "cpu_omp" || variant == "cpu_do")
        params = "threads=" + std::to_string(opt.threads > 0 ? opt.threads : omp_get_max_threads());
    if (variant == "cpu_omp") params += std::string(" schedule=") + cpu::scheduleName(opt.schedule);
    if (variant == "cpu_do") {
        char b[64];
        std::snprintf(b, sizeof b, " alpha=%g beta=%g", opt.alpha, opt.beta);
        params += b;
    }
    if (gpuVariant) params = "tg=" + std::to_string(gopt.tg);
    if (variant == "gpu_binned")
        params += " small_max=" + std::to_string(gopt.smallMax) + " large_min=" + std::to_string(gopt.largeMin);
    if (variant == "gpu_edge" || variant == "gpu_do" || variant == "hybrid")
        params += " ept=" + std::to_string(gopt.ept);
    if (variant == "gpu_do" || variant == "hybrid") {
        char b[64];
        std::snprintf(b, sizeof b, " alpha=%g beta=%g", gopt.alpha, gopt.beta);
        params += b;
    }
    if (variant == "hybrid")
        params += " T=" + tArg + " threads=" +
                  std::to_string(gopt.cpuThreads > 0 ? gopt.cpuThreads : omp_get_max_threads());

    std::printf("%s: n %llu, symmetrized m %llu (%llu undirected edges), prep %.0f ms\n", dataset.c_str(),
                (unsigned long long)g.n, (unsigned long long)g.m, (unsigned long long)(g.m / 2), prepMs);
    std::printf("bfs %s [%s], %zu source(s), %d warm-up + %d trial(s)%s\n", variant.c_str(), params.c_str(),
                sources.size(), warmup, trials, validate ? ", validating" : "");

    // GPU variants write straight into their shared output buffer.
    std::vector<uint32_t> distHost(gbfs ? 0 : g.n), ref(g.n);
    uint32_t* dist = gbfs ? gbfs->dist() : distHost.data();
    for (int w = 0; w < warmup; ++w) fn(g, toRun(sources[w % sources.size()]), dist, opt, nullptr);

    // CSV output: header written when the file is new; rows appended.
    const std::string csvPath = args.get("csv", "");
    std::ofstream csv;
    if (!csvPath.empty()) {
        bool fresh = !fileExists(csvPath);
        csv.open(csvPath, std::ios::app);
        if (fresh)
            csv << "timestamp,git_commit,chip,dataset,n,m,ordering,algo,variant,params,source,trial,"
                   "time_ms_e2e,time_ms_gpu,gteps,iters,validated,load1\n";
    }
    const std::string ts = util::timestamp(), commit = util::gitCommit(), chip = util::chipName();
    const double load1 = util::loadAverage1();

    std::vector<double> allTimes, allGpu, perSourceGteps;
    for (size_t si = 0; si < sources.size(); ++si) {
        // Reference for this source (untimed): validation target and TEPS
        // denominator. Computed per source and not kept: holding all 64
        // reference arrays costs 64*4n bytes (786 MB on Orkut) and pushed the
        // 8 GB machine into swap.
        cpu::bfsSerial(refG, sources[si], ref.data());
        uint64_t degSum = 0;
        for (uint64_t v = 0; v < refG.n; ++v) if (ref[v] != cpu::kUnreached) degSum += refG.degree(v);
        const uint64_t compEdges = degSum / 2;

        for (int w = 0; w < sourceWarmup && trials > 0; ++w) fn(g, toRun(sources[si]), dist, opt, nullptr);
        std::vector<double> times;
        for (int t = 0; t < trials; ++t) {
            util::Timer timer;
            fn(g, toRun(sources[si]), dist, opt, nullptr);
            double ms = timer.ms();
            double gpuMs = gbfs ? gbfs->lastGpuMs() : -1;
            uint64_t bad = 0;
            if (validate && !matches(dist, ref, &bad)) {
                std::fprintf(stderr, "VALIDATION FAILED: %s [%s] source %u trial %d: vertex %llu: got %u, expected %u\n",
                             variant.c_str(), ordering.c_str(), sources[si], t, (unsigned long long)bad,
                             dist[toRun((uint32_t)bad)], ref[bad]);
                return 1;
            }
            double gteps = compEdges / (ms * 1e6);
            times.push_back(ms);
            allTimes.push_back(ms);
            if (gpuMs >= 0) allGpu.push_back(gpuMs);
            if (csv.is_open()) {
                char gpuField[32] = "";
                if (gpuMs >= 0) std::snprintf(gpuField, sizeof gpuField, "%.4f", gpuMs);
                char row[1024];
                std::snprintf(row, sizeof row, "%s,%s,%s,%s,%llu,%llu,%s,bfs,%s,%s,%u,%d,%.4f,%s,%.6f,,%d,%.2f\n",
                              ts.c_str(), commit.c_str(), chip.c_str(), dataset.c_str(), (unsigned long long)g.n,
                              (unsigned long long)g.m, ordering.c_str(), variant.c_str(), params.c_str(),
                              sources[si], t, ms, gpuField, gteps, validate ? 1 : 0, load1);
                csv << row;
            }
        }
        if (trials > 0) perSourceGteps.push_back(compEdges / (median(times) * 1e6));
    }

    // Graph500 reports the harmonic mean of TEPS over sources.
    double hsum = 0;
    size_t hcount = 0;
    for (double x : perSourceGteps) if (x > 0) { hsum += 1.0 / x; ++hcount; }
    double hmean = hcount ? hcount / hsum : 0;
    if (!allTimes.empty()) {
        char gpuPart[64] = "";
        if (!allGpu.empty()) std::snprintf(gpuPart, sizeof gpuPart, " | GPU-only median %.3f", median(allGpu));
        std::printf("time ms: median %.3f, min %.3f%s | GTEPS harmonic mean %.4f%s\n", median(allTimes),
                    *std::min_element(allTimes.begin(), allTimes.end()), gpuPart, hmean,
                    validate ? " | validated OK" : "");
    }

    // --trace: one extra untimed, traced run per source.
    const std::string tracePath = args.get("trace", "");
    if (!tracePath.empty()) {
        bool fresh = !fileExists(tracePath);
        std::ofstream tr(tracePath, std::ios::app);
        if (fresh) tr << "dataset,ordering,variant,params,source,level,frontier,edges,direction,device,time_ms\n";
        for (uint32_t s : sources) {
            std::vector<util::BfsLevel> levels;
            fn(g, toRun(s), dist, opt, &levels);  // trace rows keep the original source id
            for (auto& L : levels) {
                char row[512];
                std::snprintf(row, sizeof row, "%s,%s,%s,%s,%u,%u,%llu,%llu,%c,%c,%.4f\n", dataset.c_str(),
                              ordering.c_str(), variant.c_str(), params.c_str(), s, L.level,
                              (unsigned long long)L.frontier, (unsigned long long)L.edges, L.direction,
                              L.device, L.time_ms);
                tr << row;
            }
        }
        std::printf("trace written to %s\n", tracePath.c_str());
    }

    // --dump-dist: raw uint32 distances of the (last) run, for external checks.
    const std::string dumpPath = args.get("dump-dist", "");
    if (!dumpPath.empty()) {
        fn(g, toRun(sources.back()), dist, opt, nullptr);
        std::vector<uint32_t> orig(g.n);  // back to original ids
        for (uint64_t v = 0; v < g.n; ++v) orig[v] = dist[toRun((uint32_t)v)];
        std::ofstream out(dumpPath, std::ios::binary);
        out.write(reinterpret_cast<const char*>(orig.data()), g.n * sizeof(uint32_t));
    }
    return 0;
}

// ---------------------------------------------------------------- PageRank
//
// PLAN.md 5.2: runs on the directed graph. Pull variants read the transpose
// (in-neighbors), gpu_push the stored out-CSR; out-degrees come from the
// stored graph. Undirected inputs are their own transpose (no copy).
// Fixed-iteration mode (--iters, default 20) is the benchmark metric: report
// ms per iteration. --tol runs to convergence (--iters becomes the cap,
// default 1000) and reports the iteration count. Validation (PLAN.md 7):
// cpu_serial (double) in the same mode; L1 < 1e-4, max relative error on the
// top-1000 vertices < 1e-3, sum 1 +- 1e-4.
// CSV: source empty; iters = iterations run; gteps = m * iters / time.
// Effective bandwidth (pull variants, PLAN.md Phase 8): bytes per iteration
// = 4(n+1) + 4m + 4m + 4n (row_ptr, col_idx, contrib gathers, writes).

const std::vector<std::string> kPrVariants = {"cpu_serial", "cpu_omp", "gpu_push", "gpu_pull_scalar",
                                              "gpu_pull_vector", "gpu_pull_adaptive"};

int runPr(const util::Args& args, const CSR& stored, const std::string& dataset) {
    const std::string variant = args.get("variant", "cpu_serial");
    if (std::find(kPrVariants.begin(), kPrVariants.end(), variant) == kPrVariants.end()) {
        std::fprintf(stderr, "unknown PageRank variant '%s'. Available:", variant.c_str());
        for (auto& k : kPrVariants) std::fprintf(stderr, " %s", k.c_str());
        std::fprintf(stderr, "\n");
        return 2;
    }
    const bool gpuVariant = variant.rfind("gpu_", 0) == 0;
    const double tol = args.getDouble("tol", 0);
    cpu::PrOptions copt;
    copt.tol = tol;
    copt.iters = (int)args.getInt("iters", tol > 0 ? 1000 : 20);
    copt.threads = (int)args.getInt("threads", 0);
    gpu::GpuPrOptions gopt;
    gopt.iters = copt.iters;
    gopt.tol = tol;
    gopt.tg = (uint32_t)args.getInt("tg", 256);
    gopt.smallMax = (uint32_t)args.getInt("small-max", 16);
    gopt.largeMin = (uint32_t)args.getInt("large-min", 256);

    util::Timer prep;
    const bool needIn = variant != "gpu_push";
    const bool validate = args.has("validate");
    const std::string ordering = args.get("ordering", "original");
    // --ordering: perm from the symmetrized graph (as for BFS), applied to the directed graph.
    std::vector<uint32_t> perm;
    {
        CSR sym = graph::symmetrize(stored);
        perm = orderingPerm(args, sym, dataset);
    }
    const bool permuted = !perm.empty();

    // Reference in ORIGINAL labels, computed before relabeling so the
    // original transpose is freed before the relabeled graph is built
    // (Orkut: three ~0.9 GB copies at once would exceed the 3 GB budget).
    std::vector<double> ref;
    int refIters = 0;
    if (validate) {
        CSR inO = stored.directed() ? graph::transpose(stored) : stored;
        std::vector<uint32_t> odO = cpu::outDegrees(stored);
        ref.resize(stored.n);
        refIters = cpu::pagerankSerial(inO, odO.data(), ref.data(), copt).iters;
    }

    CSR graphP = permuted ? graph::permute(stored, perm) : stored;  // what the variant runs on
    CSR in = graphP.directed() ? (needIn ? graph::transpose(graphP) : CSR{}) : graphP;
    std::vector<uint32_t> outdeg = cpu::outDegrees(graphP);
    std::unique_ptr<gpu::MetalContext> ctx;
    std::unique_ptr<gpu::GpuPageRank> gpr;
    if (gpuVariant) {
        ctx = std::make_unique<gpu::MetalContext>();
        gpr = std::make_unique<gpu::GpuPageRank>(*ctx, graphP, in, outdeg, variant == "gpu_push", needIn);
        if (variant == "gpu_pull_adaptive") gpr->prepareAdaptive(gopt.smallMax, gopt.largeMin);
        in = CSR{};      // the GPU has its own copies; drop the host ones
        graphP = CSR{};
    }
    const double prepMs = prep.ms();
    const uint64_t n = stored.n, m = stored.m;

    std::string params = tol > 0 ? "tol=" + std::to_string(tol).substr(0, 8) + " cap=" + std::to_string(copt.iters)
                                 : "iters=" + std::to_string(copt.iters);
    if (variant == "cpu_omp")
        params += " threads=" + std::to_string(copt.threads > 0 ? copt.threads : omp_get_max_threads());
    if (gpuVariant) params += " tg=" + std::to_string(gopt.tg);
    if (variant == "gpu_pull_adaptive")
        params += " small_max=" + std::to_string(gopt.smallMax) + " large_min=" + std::to_string(gopt.largeMin);
    std::printf("%s: n %llu, m %llu (directed=%d), ordering %s, prep %.0f ms\npr %s [%s]\n", dataset.c_str(),
                (unsigned long long)n, (unsigned long long)m, (int)stored.directed(), ordering.c_str(), prepMs,
                variant.c_str(), params.c_str());

    std::vector<double> rcpu(n);
    // Runs the variant once; returns {iterations, gpu ms (or -1)}; sets lastDelta.
    double lastDelta = -1;
    bool floorStop = false;
    auto runOnce = [&]() -> std::pair<int, double> {
        if (!gpuVariant) {
            cpu::PrResult r = variant == "cpu_serial" ? cpu::pagerankSerial(in, outdeg.data(), rcpu.data(), copt)
                                                      : cpu::pagerankOmp(in, outdeg.data(), rcpu.data(), copt);
            lastDelta = r.lastDelta;
            return {r.iters, -1};
        }
        gpu::GpuPrResult r = gpr->run(gpu::parsePrVariant(variant), gopt);
        lastDelta = r.lastDelta;
        floorStop = r.floorStop;
        return {r.iters, r.gpuMs};
    };
    // Ranks in ORIGINAL ids (rank of original vertex v = result[perm[v]]).
    auto originalRanks = [&]() {
        std::vector<double> out(n);
        for (uint64_t v = 0; v < n; ++v) {
            uint64_t r = permuted ? perm[v] : v;
            out[v] = gpuVariant ? gpr->ranks()[r] : rcpu[r];
        }
        return out;
    };
    auto check = [&](const std::vector<double>& reference) {
        return cpu::checkRanks(reference, originalRanks().data());
    };

    const int warmup = (int)args.getInt("warmup", 2), trials = (int)args.getInt("trials", 5);
    for (int w = 0; w < warmup; ++w) runOnce();

    const std::string csvPath = args.get("csv", "");
    std::ofstream csv;
    if (!csvPath.empty()) {
        bool fresh = !fileExists(csvPath);
        csv.open(csvPath, std::ios::app);
        if (fresh)
            csv << "timestamp,git_commit,chip,dataset,n,m,ordering,algo,variant,params,source,trial,"
                   "time_ms_e2e,time_ms_gpu,gteps,iters,validated,load1\n";
    }
    const std::string ts = util::timestamp(), commit = util::gitCommit(), chip = util::chipName();
    const double load1 = util::loadAverage1();

    std::vector<double> times, gpuTimes;
    int iters = 0;
    cpu::PrCheck worst;
    for (int t = 0; t < trials; ++t) {
        util::Timer timer;
        auto [it, gpuMs] = runOnce();
        double ms = timer.ms();
        iters = it;
        if (validate) {
            cpu::PrCheck c = check(ref);
            if (!c.ok()) {
                std::fprintf(stderr, "VALIDATION FAILED: %s trial %d: L1 %.3g, max rel (top-1000) %.3g, sum %.8f\n",
                             variant.c_str(), t, c.l1, c.maxRelTop, c.sum);
                return 1;
            }
            if (c.l1 >= worst.l1) worst = c;
        }
        times.push_back(ms);
        if (gpuMs >= 0) gpuTimes.push_back(gpuMs);
        if (csv.is_open()) {
            char gpuField[32] = "";
            if (gpuMs >= 0) std::snprintf(gpuField, sizeof gpuField, "%.4f", gpuMs);
            char row[1024];
            std::snprintf(row, sizeof row, "%s,%s,%s,%s,%llu,%llu,%s,pr,%s,%s,,%d,%.4f,%s,%.6f,%d,%d,%.2f\n",
                          ts.c_str(), commit.c_str(), chip.c_str(), dataset.c_str(), (unsigned long long)n,
                          (unsigned long long)m, ordering.c_str(), variant.c_str(), params.c_str(), t, ms,
                          gpuField, (double)m * it / (ms * 1e6), it, validate ? 1 : 0, load1);
            csv << row;
        }
    }
    if (!times.empty()) {
        const double med = median(times), perIt = med / std::max(1, iters);
        std::printf("time ms: median %.3f, min %.3f | %d iterations, %.4f ms/iteration",
                    med, *std::min_element(times.begin(), times.end()), iters, perIt);
        if (!gpuTimes.empty()) std::printf(" (GPU-only %.4f)", median(gpuTimes) / std::max(1, iters));
        if (tol > 0)
            std::printf(" | final L1 change %.3g%s", lastDelta,
                        lastDelta < tol ? "" : floorStop ? " (stopped at float precision floor)" : " (NOT converged)");
        if (variant != "gpu_push" && variant != "cpu_serial") {
            // PLAN.md Phase 8 effective-bandwidth estimate, from the e2e time per iteration.
            double bytes = 4.0 * (n + 1) + 8.0 * m + 4.0 * n;
            std::printf(" | eff. %.1f GB/s, %.3f Ggathers/s", bytes / (perIt * 1e6), m / (perIt * 1e6));
        }
        if (validate)
            std::printf(" | validated OK (ref %d iters; worst L1 %.2e, top-1000 rel %.2e, sum-1 %.1e)", refIters,
                        worst.l1, worst.maxRelTop, worst.sum - 1.0);
        std::printf("\n");
    }

    const std::string dumpPath = args.get("dump-ranks", "");
    if (!dumpPath.empty()) {
        if (trials == 0) runOnce();
        std::vector<double> outRanks = originalRanks();
        std::ofstream o(dumpPath, std::ios::binary);
        o.write(reinterpret_cast<const char*>(outRanks.data()), n * sizeof(double));
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    util::Args args(argc, argv);
    if (!args.has("graph")) {
        std::fprintf(stderr,
                     "usage: gbench --graph <file.bin | synth:NAME> --algo bfs --variant V [options]\n"
                     "       (see the comment at the top of src/bench/main.cpp)\n");
        return 2;
    }
    try {
        std::string dataset;
        CSR g = loadGraph(args.get("graph"), dataset);
        if (args.has("save-graph")) {
            graph::save(g, args.get("save-graph"));
            return 0;
        }
        const std::string algo = args.get("algo", "bfs");
        if (algo == "bfs") return runBfs(args, g, dataset);
        if (algo == "pr") return runPr(args, g, dataset);
        std::fprintf(stderr, "unknown --algo '%s' (bfs | pr)\n", algo.c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gbench: %s\n", e.what());
        return 1;
    }
}
