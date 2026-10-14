// gstats: dataset statistics.
//   gstats <graph.bin> [--csv results/dataset_stats.csv] [--hist results/degree_hist.csv]
// Loads and fully validates the graph, then reports n, m, degree stats, a
// log2 degree histogram, connected components and an approximate diameter
// (max BFS depth from 10 random sources) on the symmetrized graph.
#include "cpu/bfs_cpu.hpp"
#include "graph/csr.hpp"
#include "graph/transform.hpp"
#include "util/args.hpp"
#include "util/csv.hpp"
#include "util/rng.hpp"
#include "util/timer.hpp"

#include <algorithm>
#include <cstdio>
#include <sys/stat.h>

using graph::CSR;

// Bucket 0 holds degree 0; bucket k >= 1 holds degrees in [2^(k-1), 2^k).
static std::vector<uint64_t> log2Histogram(const std::vector<uint32_t>& deg) {
    std::vector<uint64_t> h(34, 0);
    for (uint32_t d : deg) h[d == 0 ? 0 : 32 - __builtin_clz(d)]++;
    while (h.size() > 1 && h.back() == 0) h.pop_back();
    return h;
}

static void histRows(std::vector<std::string>& rows, const std::string& name, const char* kind,
                     const std::vector<uint32_t>& deg) {
    auto h = log2Histogram(deg);
    for (size_t k = 0; k < h.size(); ++k) {
        uint64_t lo = k == 0 ? 0 : 1ull << (k - 1), hi = k == 0 ? 0 : (1ull << k) - 1;
        char row[256];
        std::snprintf(row, sizeof row, "%s,%s,%zu,%llu,%llu,%llu", name.c_str(), kind, k,
                      (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)h[k]);
        rows.push_back(row);
    }
}

int main(int argc, char** argv) {
    util::Args args(argc, argv);
    if (args.positional().size() != 1) {
        std::fprintf(stderr, "usage: gstats <graph.bin> [--csv path] [--hist path]\n");
        return 2;
    }
    const std::string path = args.positional()[0], name = util::stem(path);
    util::Timer t;
    CSR g;
    try {
        g = graph::load(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gstats: %s\n", e.what());
        return 1;
    }
    std::string err = graph::validate(g);
    if (!err.empty()) { std::fprintf(stderr, "gstats: %s INVALID: %s\n", path.c_str(), err.c_str()); return 1; }
    double loadMs = t.ms();

    t.reset();
    CSR s = graph::symmetrize(g);
    err = graph::validate(s);
    if (!err.empty()) { std::fprintf(stderr, "gstats: symmetrized graph INVALID: %s\n", err.c_str()); return 1; }
    double symMs = t.ms();

    // Degrees of the stored graph (out-degree if directed) and of the symmetrized graph.
    std::vector<uint32_t> outDeg(g.n), symDeg(g.n), inDeg;
    for (uint64_t v = 0; v < g.n; ++v) { outDeg[v] = g.degree(v); symDeg[v] = s.degree(v); }
    if (g.directed()) {
        inDeg.assign(g.n, 0);
        for (uint64_t i = 0; i < g.m; ++i) inDeg[g.col_idx[i]]++;
    }
    uint32_t minDeg = g.n ? *std::min_element(outDeg.begin(), outDeg.end()) : 0;
    uint32_t maxDeg = g.n ? *std::max_element(outDeg.begin(), outDeg.end()) : 0;
    uint32_t maxIn = inDeg.empty() ? maxDeg : *std::max_element(inDeg.begin(), inDeg.end());
    uint64_t isolated = std::count(symDeg.begin(), symDeg.end(), 0u);
    double meanDeg = g.n ? (double)g.m / g.n : 0;

    // Connected components of the symmetrized graph, by repeated BFS labeling.
    t.reset();
    std::vector<uint32_t> comp(s.n, cpu::kUnreached), queue(s.n);
    uint64_t numCC = 0, largestCC = 0;
    for (uint64_t r = 0; r < s.n; ++r) {
        if (comp[r] != cpu::kUnreached) continue;
        size_t head = 0, tail = 0;
        comp[r] = (uint32_t)numCC;
        queue[tail++] = (uint32_t)r;
        while (head < tail) {
            uint32_t u = queue[head++];
            for (const uint32_t* p = s.begin(u); p != s.end(u); ++p)
                if (comp[*p] == cpu::kUnreached) { comp[*p] = (uint32_t)numCC; queue[tail++] = *p; }
        }
        largestCC = std::max<uint64_t>(largestCC, tail);
        ++numCC;
    }
    double ccMs = t.ms();

    // Approximate diameter: max finite BFS depth from 10 random non-isolated sources.
    t.reset();
    util::Rng rng(1);
    uint32_t diam = 0;
    if (isolated < s.n) {
        for (int k = 0; k < 10; ++k) {
            uint32_t src;
            do { src = (uint32_t)rng.below(s.n); } while (symDeg[src] == 0);
            auto dist = cpu::bfsSerial(s, src);
            for (uint32_t d : dist) if (d != cpu::kUnreached) diam = std::max(diam, d);
        }
    }
    double diamMs = t.ms();

    struct stat st;
    double fileMb = stat(path.c_str(), &st) == 0 ? st.st_size / (1024.0 * 1024.0) : 0;

    std::printf("%s (%s)\n", name.c_str(), g.directed() ? "directed" : "undirected");
    std::printf("  n %llu, m %llu (stored entries), file %.1f MB\n", (unsigned long long)g.n,
                (unsigned long long)g.m, fileMb);
    std::printf("  symmetrized: %llu entries = %llu undirected edges\n", (unsigned long long)s.m,
                (unsigned long long)(s.m / 2));
    std::printf("  %sdegree min/mean/max %u / %.2f / %u%s\n", g.directed() ? "out-" : "", minDeg,
                meanDeg, maxDeg, g.directed() ? (", max in-degree " + std::to_string(maxIn)).c_str() : "");
    std::printf("  isolated vertices (symmetrized) %llu\n", (unsigned long long)isolated);
    std::printf("  connected components %llu, largest %llu (%.1f%%)\n", (unsigned long long)numCC,
                (unsigned long long)largestCC, g.n ? 100.0 * largestCC / g.n : 0.0);
    std::printf("  approx diameter (10 BFS) %u\n", diam);
    std::printf("  timings: load+validate %.0f ms, symmetrize %.0f ms, CC %.0f ms, 10 BFS %.0f ms\n",
                loadMs, symMs, ccMs, diamMs);

    std::string csv = args.get("csv", "results/dataset_stats.csv");
    if (!csv.empty()) {
        char row[512];
        std::snprintf(row, sizeof row, "%s,%llu,%llu,%d,%llu,%u,%.3f,%u,%u,%llu,%llu,%llu,%u,%.1f",
                      name.c_str(), (unsigned long long)g.n, (unsigned long long)g.m, g.directed() ? 1 : 0,
                      (unsigned long long)(s.m / 2), minDeg, meanDeg, maxDeg, maxIn,
                      (unsigned long long)isolated, (unsigned long long)numCC, (unsigned long long)largestCC,
                      diam, fileMb);
        util::csvUpsert(csv,
                        "dataset,n,m,directed,undirected_edges,min_deg,mean_deg,max_deg,max_in_deg,"
                        "isolated,num_cc,largest_cc,approx_diameter,file_mb",
                        name, {row});
    }
    std::string hist = args.get("hist", "results/degree_hist.csv");
    if (!hist.empty()) {
        std::vector<std::string> rows;
        histRows(rows, name, g.directed() ? "out" : "deg", outDeg);
        if (g.directed()) { histRows(rows, name, "in", inDeg); histRows(rows, name, "sym", symDeg); }
        util::csvUpsert(hist, "dataset,kind,bucket,lo,hi,count", name, rows);
    }
    return 0;
}
