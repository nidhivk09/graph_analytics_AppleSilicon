// gbench: benchmark + validation CLI for every BFS (and later PageRank) variant.
//
//   gbench --graph data/soc-LiveJournal1.bin --algo bfs --variant cpu_omp
//          [--threads 8] [--schedule static|dynamic,64|guided] [--alpha 15 --beta 18]
//          [--sources 64 | --source S] [--seed 1] [--any-component] [--warmup 2] [--trials 5]
//          [--validate] [--trace trace.csv] [--csv results/bfs.csv] [--ordering original]
//   gbench --graph synth:star100k ...         use a generated test graph
//   gbench --graph X --save-graph out.bin     write the (stored, unsymmetrized) graph
//   gbench --graph X --variant V --source S --dump-dist out.u32   raw uint32 distances
//
// Methodology (PLAN.md 8): BFS runs on the symmetrized graph; sources are
// random vertices with degree > 0 in the largest connected component, from a
// fixed seed (same for every variant);
// TEPS denominator = undirected edges in the source's component, computed
// from the cpu_serial result; 2 warm-up runs, then `trials` timed runs per
// source. Summary: harmonic mean of per-source GTEPS (from the median time)
// and the median / min time over all runs.
#include "cpu/bfs_cpu.hpp"
#include "graph/csr.hpp"
#include "graph/generators.hpp"
#include "graph/transform.hpp"
#include "util/args.hpp"
#include "util/rng.hpp"
#include "util/sysinfo.hpp"
#include "util/timer.hpp"

#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <sys/stat.h>

using graph::CSR;

namespace {

using BfsFn = std::function<void(const CSR&, uint32_t, uint32_t*, const cpu::BfsOptions&,
                                 std::vector<util::BfsLevel>*)>;

const std::map<std::string, BfsFn>& bfsVariants() {
    static const std::map<std::string, BfsFn> m = {
        {"cpu_serial", [](const CSR& g, uint32_t s, uint32_t* d, const cpu::BfsOptions&,
                          std::vector<util::BfsLevel>*) { cpu::bfsSerial(g, s, d); }},
        {"cpu_omp", cpu::bfsOmp},
        {"cpu_do", cpu::bfsDO},
    };
    return m;
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

int runBfs(const util::Args& args, const CSR& stored, const std::string& dataset) {
    const std::string variant = args.get("variant", "cpu_serial");
    auto it = bfsVariants().find(variant);
    if (it == bfsVariants().end()) {
        std::fprintf(stderr, "unknown BFS variant '%s'. Available:", variant.c_str());
        for (auto& [k, v] : bfsVariants()) std::fprintf(stderr, " %s", k.c_str());
        std::fprintf(stderr, "\n");
        return 2;
    }
    const BfsFn& fn = it->second;

    cpu::BfsOptions opt;
    opt.threads = (int)args.getInt("threads", 0);
    opt.schedule = cpu::parseSchedule(args.get("schedule", "dynamic,64"));
    opt.alpha = args.getDouble("alpha", 15.0);
    opt.beta = args.getDouble("beta", 18.0);

    // Transform cost is outside every timed region (PLAN 4.4).
    util::Timer prep;
    CSR g = graph::symmetrize(stored);
    double prepMs = prep.ms();

    std::vector<uint32_t> sources;
    if (args.has("source")) sources = {(uint32_t)args.getInt("source", 0)};
    else sources = pickSources(g, (int)args.getInt("sources", 64), (uint64_t)args.getInt("seed", 1),
                                args.has("any-component"));
    if (sources.empty()) { std::fprintf(stderr, "no vertex with degree > 0\n"); return 1; }
    for (uint32_t s : sources)
        if (s >= g.n) { std::fprintf(stderr, "source %u out of range\n", s); return 2; }

    const int warmup = (int)args.getInt("warmup", 2), trials = (int)args.getInt("trials", 5);
    const bool validate = args.has("validate");
    const std::string ordering = args.get("ordering", "original");

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

    std::printf("%s: n %llu, symmetrized m %llu (%llu undirected edges), prep %.0f ms\n", dataset.c_str(),
                (unsigned long long)g.n, (unsigned long long)g.m, (unsigned long long)(g.m / 2), prepMs);
    std::printf("bfs %s [%s], %zu source(s), %d warm-up + %d trial(s)%s\n", variant.c_str(), params.c_str(),
                sources.size(), warmup, trials, validate ? ", validating" : "");

    std::vector<uint32_t> dist(g.n), ref(g.n);
    for (int w = 0; w < warmup; ++w) fn(g, sources[w % sources.size()], dist.data(), opt, nullptr);

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

    std::vector<double> allTimes, perSourceGteps;
    for (size_t si = 0; si < sources.size(); ++si) {
        // Reference for this source (untimed): validation target and TEPS
        // denominator. Computed per source and not kept: holding all 64
        // reference arrays costs 64*4n bytes (786 MB on Orkut) and pushed the
        // 8 GB machine into swap.
        cpu::bfsSerial(g, sources[si], ref.data());
        uint64_t degSum = 0;
        for (uint64_t v = 0; v < g.n; ++v) if (ref[v] != cpu::kUnreached) degSum += g.degree(v);
        const uint64_t compEdges = degSum / 2;

        std::vector<double> times;
        for (int t = 0; t < trials; ++t) {
            util::Timer timer;
            fn(g, sources[si], dist.data(), opt, nullptr);
            double ms = timer.ms();
            if (validate && dist != ref) {
                for (uint64_t v = 0; v < g.n; ++v)
                    if (dist[v] != ref[v]) {
                        std::fprintf(stderr, "VALIDATION FAILED: %s source %u trial %d: dist[%llu] = %u, expected %u\n",
                                     variant.c_str(), sources[si], t, (unsigned long long)v, dist[v], ref[v]);
                        break;
                    }
                return 1;
            }
            double gteps = compEdges / (ms * 1e6);
            times.push_back(ms);
            allTimes.push_back(ms);
            if (csv.is_open()) {
                char row[1024];
                std::snprintf(row, sizeof row, "%s,%s,%s,%s,%llu,%llu,%s,bfs,%s,%s,%u,%d,%.4f,,%.6f,,%d,%.2f\n",
                              ts.c_str(), commit.c_str(), chip.c_str(), dataset.c_str(), (unsigned long long)g.n,
                              (unsigned long long)g.m, ordering.c_str(), variant.c_str(), params.c_str(),
                              sources[si], t, ms, gteps, validate ? 1 : 0, load1);
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
    if (!allTimes.empty())
        std::printf("time ms: median %.3f, min %.3f | GTEPS harmonic mean %.4f%s\n", median(allTimes),
                    *std::min_element(allTimes.begin(), allTimes.end()), hmean,
                    validate ? " | validated OK" : "");

    // --trace: one extra untimed, traced run per source.
    const std::string tracePath = args.get("trace", "");
    if (!tracePath.empty()) {
        bool fresh = !fileExists(tracePath);
        std::ofstream tr(tracePath, std::ios::app);
        if (fresh) tr << "dataset,ordering,variant,params,source,level,frontier,edges,direction,device,time_ms\n";
        for (uint32_t s : sources) {
            std::vector<util::BfsLevel> levels;
            fn(g, s, dist.data(), opt, &levels);
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
        fn(g, sources.back(), dist.data(), opt, nullptr);
        std::ofstream out(dumpPath, std::ios::binary);
        out.write(reinterpret_cast<const char*>(dist.data()), dist.size() * sizeof(uint32_t));
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
        std::fprintf(stderr, "unknown --algo '%s' (bfs; pr arrives in Phase 8)\n", algo.c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gbench: %s\n", e.what());
        return 1;
    }
}
