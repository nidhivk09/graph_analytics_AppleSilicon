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

// Out-degrees of a CSR (the original, directed graph).
std::vector<uint32_t> outDegrees(const graph::CSR& g);

} // namespace cpu
