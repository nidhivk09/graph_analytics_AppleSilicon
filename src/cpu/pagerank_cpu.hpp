#pragma once
// CPU PageRank (PLAN.md 5.2) and the rank comparison used for validation.
//
// Definition (shared by every variant): d = 0.85, r0 = 1/n,
//   r_new[v] = (1-d)/n + d * ( sum_{u in in(v)} r[u]/outdeg[u] + dangling/n )
// where dangling = total rank on vertices with outdeg 0. Pull-based: `in` is
// the transpose CSR (in-neighbor lists), `outdeg` the out-degrees of the
// original graph. The CPU variants use double (the reference precision).
#include "graph/csr.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace cpu {

struct PrOptions {
    double damping = 0.85;
    int iters = 20;        // fixed-iteration mode; in tol mode, the iteration cap
    double tol = 0;        // > 0: stop when the L1 change of an iteration is < tol
    int threads = 0;       // cpu_omp: 0 = OpenMP default
};

struct PrResult {
    int iters = 0;         // iterations performed
    double lastDelta = 0;  // L1 change of the last iteration
};

// Ranks are written to r (n entries).
PrResult pagerankSerial(const graph::CSR& in, const uint32_t* outdeg, double* r, const PrOptions& opt);
PrResult pagerankOmp(const graph::CSR& in, const uint32_t* outdeg, double* r, const PrOptions& opt);

// PLAN.md 7 acceptance: L1(ref - got) < 1e-4, max relative error over the
// reference's top-1000 vertices < 1e-3, ranks sum to 1 +- 1e-4.
struct PrCheck {
    double l1 = 0, maxRelTop = 0, sum = 0;
    bool ok() const { return l1 < 1e-4 && maxRelTop < 1e-3 && std::fabs(sum - 1.0) < 1e-4; }
};

template <class T>
PrCheck checkRanks(const std::vector<double>& ref, const T* got) {
    PrCheck c;
    const size_t n = ref.size();
    for (size_t v = 0; v < n; ++v) {
        c.l1 += std::fabs(ref[v] - (double)got[v]);
        c.sum += (double)got[v];
    }
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    size_t k = std::min<size_t>(1000, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](uint32_t a, uint32_t b) { return ref[a] > ref[b]; });
    for (size_t i = 0; i < k; ++i) {
        uint32_t v = idx[i];
        c.maxRelTop = std::max(c.maxRelTop, std::fabs(ref[v] - (double)got[v]) / ref[v]);
    }
    return c;
}

// Acceptance for gpu_pull_gs in --tol mode (Phase 12). The normal check compares a variant
// with cpu_serial run the same way, which follows the *same* Jacobi
// trajectory, so it measures only float rounding. A Gauss-Seidel run is a
// different approximation of the fixed point (after k sweeps, or when its
// per-sweep change drops below tol), so that comparison would fail for the
// wrong reason. Instead both are compared with the converged reference
// (tol 1e-10): the GS result passes if every PLAN.md 7 metric is within its
// tolerance or no worse than the Jacobi reference's own distance from
// convergence, i.e. GS must be at least as accurate as the computation it
// replaces. (Fixed-iteration mode is checked differently, see gbench runPr.) Two details:
//  - Sum: Jacobi with dangling redistribution conserves the total exactly,
//    GS (dangling term from the start of the sweep) does not, so "no worse
//    sum" would be unreachable. For any vector |sum - 1| <= L1 distance to
//    the true ranks, so the sum gets the same budget as L1.
//  - Top-1000: when neither method has converged at some vertex (roadNet-CA
//    after 20 iterations: both 0.022463 at the same vertex), the float GPU
//    value can exceed the double reference's error by rounding alone, so a
//    relative slack of 1e-4 (~1000 float ulps) is allowed on that bound.
inline bool gsNoWorseThanJacobi(const PrCheck& gs, const PrCheck& jacobi) {
    const double l1Budget = std::max(1e-4, jacobi.l1);
    return gs.l1 <= l1Budget && std::fabs(gs.sum - 1.0) <= l1Budget &&
           gs.maxRelTop <= std::max(1e-3, jacobi.maxRelTop * (1 + 1e-4));
}

// Out-degrees of a CSR (the original, directed graph).
std::vector<uint32_t> outDegrees(const graph::CSR& g);

} // namespace cpu
