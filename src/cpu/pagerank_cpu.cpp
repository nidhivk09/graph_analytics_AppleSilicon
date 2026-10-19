#include "cpu/pagerank_cpu.hpp"

#include <omp.h>
#include <cmath>
#include <vector>

namespace cpu {

std::vector<uint32_t> outDegrees(const graph::CSR& g) {
    std::vector<uint32_t> d(g.n);
    for (uint64_t v = 0; v < g.n; ++v) d[v] = g.degree((uint32_t)v);
    return d;
}

namespace {

// One implementation for both variants; `threads` = 1 runs without OpenMP
// overhead (serial reference), otherwise dynamic scheduling over vertices
// (in-degree is skewed on social graphs, so static chunks are unbalanced).
PrResult pagerank(const graph::CSR& in, const uint32_t* outdeg, double* r, const PrOptions& opt,
                  bool parallel) {
    const uint64_t n = in.n;
    if (n == 0) return {};
    const int nt = parallel ? (opt.threads > 0 ? opt.threads : omp_get_max_threads()) : 1;
    const double d = opt.damping, base = (1.0 - d) / n;
    std::vector<double> contrib(n), next(n);
    for (uint64_t v = 0; v < n; ++v) r[v] = 1.0 / n;

    PrResult res;
    const int cap = opt.iters;
    for (int it = 0; it < cap; ++it) {
        double dangling = 0;
#pragma omp parallel for num_threads(nt) schedule(static) reduction(+ : dangling) if (parallel)
        for (uint64_t u = 0; u < n; ++u) {
            if (outdeg[u]) contrib[u] = r[u] / outdeg[u];
            else { contrib[u] = 0; dangling += r[u]; }
        }
        const double add = base + d * dangling / n;
        double delta = 0;
#pragma omp parallel for num_threads(nt) schedule(dynamic, 256) reduction(+ : delta) if (parallel)
        for (uint64_t v = 0; v < n; ++v) {
            double s = 0;
            for (const uint32_t* p = in.begin((uint32_t)v); p != in.end((uint32_t)v); ++p) s += contrib[*p];
            next[v] = add + d * s;
            delta += std::fabs(next[v] - r[v]);
        }
        for (uint64_t v = 0; v < n; ++v) r[v] = next[v];
        res.iters = it + 1;
        res.lastDelta = delta;
        if (opt.tol > 0 && delta < opt.tol) break;
    }
    return res;
}

} // namespace

PrResult pagerankSerial(const graph::CSR& in, const uint32_t* outdeg, double* r, const PrOptions& opt) {
    return pagerank(in, outdeg, r, opt, false);
}

PrResult pagerankOmp(const graph::CSR& in, const uint32_t* outdeg, double* r, const PrOptions& opt) {
    return pagerank(in, outdeg, r, opt, true);
}

} // namespace cpu
