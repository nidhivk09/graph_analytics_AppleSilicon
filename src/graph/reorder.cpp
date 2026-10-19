#include "graph/reorder.hpp"

#include "util/rng.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace graph::reorder {

namespace {

std::vector<uint32_t> identity(uint64_t n) {
    std::vector<uint32_t> p(n);
    std::iota(p.begin(), p.end(), 0u);
    return p;
}

// Fisher-Yates with the project RNG: same permutation on every run.
std::vector<uint32_t> randomPerm(uint64_t n, uint64_t seed) {
    std::vector<uint32_t> p = identity(n);
    util::Rng rng(seed);
    for (uint64_t i = n; i > 1; --i) std::swap(p[i - 1], p[rng.below(i)]);
    return p;
}

// Hub sort: new ids in order of descending degree (stable: ties keep
// original order). Puts the hubs, whose ranks/dist every edge touches most,
// together at the front of every per-vertex array.
std::vector<uint32_t> degreePerm(const CSR& g) {
    std::vector<uint32_t> order = identity(g.n);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return g.degree(a) > g.degree(b); });
    std::vector<uint32_t> perm(g.n);
    for (uint64_t i = 0; i < g.n; ++i) perm[order[i]] = (uint32_t)i;
    return perm;
}

// BFS from `start` over unvisited vertices; appends visit order to `out`.
// Neighbors are visited in ascending degree (Cuthill-McKee). `mark` is the
// visited flag; returns the index in `out` where the last level begins.
size_t cmBfs(const CSR& g, uint32_t start, std::vector<uint8_t>& mark, std::vector<uint32_t>& out,
             std::vector<uint32_t>& scratch) {
    size_t head = out.size(), lastLevel = head;
    out.push_back(start);
    mark[start] = 1;
    while (head < out.size()) {
        size_t levelEnd = out.size();
        lastLevel = head;
        for (; head < levelEnd; ++head) {
            uint32_t u = out[head];
            scratch.clear();
            for (const uint32_t* p = g.begin(u); p != g.end(u); ++p)
                if (!mark[*p]) { mark[*p] = 1; scratch.push_back(*p); }
            std::sort(scratch.begin(), scratch.end(), [&](uint32_t a, uint32_t b) {
                return g.degree(a) != g.degree(b) ? g.degree(a) < g.degree(b) : a < b;
            });
            out.insert(out.end(), scratch.begin(), scratch.end());
        }
    }
    return lastLevel;
}

// Reverse Cuthill-McKee. Per connected component: start at its lowest-degree
// unvisited vertex, refine once to a pseudo-peripheral start (the lowest-
// degree vertex of the farthest BFS level, the usual GPS-style heuristic),
// run the degree-ordered BFS from there; finally reverse the whole order.
std::vector<uint32_t> rcmPerm(const CSR& g) {
    const uint64_t n = g.n;
    std::vector<uint32_t> byDegree = identity(n);
    std::stable_sort(byDegree.begin(), byDegree.end(),
                     [&](uint32_t a, uint32_t b) { return g.degree(a) < g.degree(b); });
    std::vector<uint8_t> mark(n, 0), probe(n, 0);
    std::vector<uint32_t> order, tmp, scratch;
    order.reserve(n);
    for (uint32_t s : byDegree) {
        if (mark[s]) continue;
        // Refinement pass with a separate marking (`probe`). Components are
        // disjoint, so probe never needs clearing between components.
        tmp.clear();
        size_t last = cmBfs(g, s, probe, tmp, scratch);
        uint32_t start = tmp[last];
        for (size_t i = last; i < tmp.size(); ++i)
            if (g.degree(tmp[i]) < g.degree(start)) start = tmp[i];
        cmBfs(g, start, mark, order, scratch);
    }
    std::vector<uint32_t> perm(n);
    for (uint64_t i = 0; i < n; ++i) perm[order[i]] = (uint32_t)(n - 1 - i);  // reversed
    return perm;
}

} // namespace

std::vector<uint32_t> compute(const CSR& sym, const std::string& ordering, uint64_t seed) {
    if (ordering == "original") return identity(sym.n);
    if (ordering == "random") return randomPerm(sym.n, seed);
    if (ordering == "degree") return degreePerm(sym);
    if (ordering == "rcm") return rcmPerm(sym);
    throw std::runtime_error("unknown ordering '" + ordering + "' (original|random|degree|rcm)");
}

bool isPermutation(const std::vector<uint32_t>& p) {
    std::vector<uint8_t> seen(p.size(), 0);
    for (uint32_t x : p) {
        if (x >= p.size() || seen[x]) return false;
        seen[x] = 1;
    }
    return true;
}

Locality locality(const CSR& g) {
    double gap = 0;
    uint64_t same = 0;
    for (uint64_t u = 0; u < g.n; ++u)
        for (const uint32_t* p = g.begin((uint32_t)u); p != g.end((uint32_t)u); ++p) {
            gap += u > *p ? (double)(u - *p) : (double)(*p - u);
            same += (u >> 6) == (*p >> 6);
        }
    Locality l;
    if (g.m) {
        l.meanGap = gap / g.m;
        l.sameBlock64 = (double)same / g.m;
    }
    return l;
}

} // namespace graph::reorder
