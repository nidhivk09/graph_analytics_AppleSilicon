#include "graph/transform.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace graph {

CSR transpose(const CSR& g) {
    const uint64_t n = g.n;
    std::vector<uint32_t> rp(n + 1, 0);
    for (uint64_t i = 0; i < g.m; ++i) rp[g.col_idx[i] + 1]++;
    for (uint64_t v = 0; v < n; ++v) rp[v + 1] += rp[v];

    // Serial scatter in increasing source order => each in-list is sorted.
    // (A parallel scatter would need atomics and a re-sort; transforms are
    // outside every timed region, so simple wins.)
    std::vector<uint32_t> ci(g.m);
    std::vector<uint32_t> cur(rp.begin(), rp.end() - 1);
    for (uint64_t u = 0; u < n; ++u)
        for (uint32_t i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i)
            ci[cur[g.col_idx[i]]++] = (uint32_t)u;
    return CSR::fromVectors(std::move(rp), std::move(ci), g.flags);
}

CSR symmetrize(const CSR& g) {
    if (!g.directed()) return g;
    const int64_t n = (int64_t)g.n;
    CSR t = transpose(g);

    // Per vertex, merge two sorted lists (out and in) without duplicates.
    // Pass 1 counts merged sizes, pass 2 writes; both parallel over vertices.
    auto mergeInto = [&](int64_t v, uint32_t* out) -> uint32_t {
        const uint32_t *a = g.begin(v), *ae = g.end(v), *b = t.begin(v), *be = t.end(v);
        uint32_t k = 0;
        while (a != ae || b != be) {
            uint32_t x;
            if (b == be || (a != ae && *a < *b)) x = *a++;
            else if (a == ae || *b < *a) x = *b++;
            else { x = *a++; ++b; }  // equal: reciprocal edge, keep once
            if (out) out[k] = x;
            ++k;
        }
        return k;
    };

    std::vector<uint32_t> rp(n + 1, 0);
#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t v = 0; v < n; ++v) rp[v + 1] = mergeInto(v, nullptr);
    for (int64_t v = 0; v < n; ++v) rp[v + 1] += rp[v];

    std::vector<uint32_t> ci(rp[n]);
#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t v = 0; v < n; ++v) mergeInto(v, ci.data() + rp[v]);

    return CSR::fromVectors(std::move(rp), std::move(ci), kFlagSorted | kFlagDedup);
}

CSR permute(const CSR& g, const std::vector<uint32_t>& perm) {
    const uint64_t n = g.n;
    if (perm.size() != n) throw std::runtime_error("permute: perm size != n");
    std::vector<uint32_t> inv(n);
    for (uint64_t v = 0; v < n; ++v) inv[perm[v]] = (uint32_t)v;
    std::vector<uint32_t> rp(n + 1, 0);
    for (uint64_t nv = 0; nv < n; ++nv) rp[nv + 1] = rp[nv] + g.degree(inv[nv]);
    std::vector<uint32_t> ci(g.m);
    // Each new row is independent: relabel and sort in parallel.
#pragma omp parallel for schedule(dynamic, 1024)
    for (uint64_t nv = 0; nv < n; ++nv) {
        uint32_t* out = ci.data() + rp[nv];
        uint32_t k = 0;
        for (const uint32_t* p = g.begin(inv[nv]); p != g.end(inv[nv]); ++p) out[k++] = perm[*p];
        std::sort(out, out + k);
    }
    return CSR::fromVectors(std::move(rp), std::move(ci), g.flags);
}

} // namespace graph
