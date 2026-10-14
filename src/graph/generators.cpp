#include "graph/generators.hpp"
#include "util/rng.hpp"

namespace graph::gen {

using EdgeList = std::vector<std::pair<uint32_t, uint32_t>>;

CSR path(uint32_t n) {
    EdgeList e;
    for (uint32_t i = 0; i + 1 < n; ++i) e.push_back({i, i + 1});
    return fromEdgeList(n, e, false);
}

CSR star(uint32_t leaves) {
    EdgeList e;
    for (uint32_t i = 1; i <= leaves; ++i) e.push_back({0, i});
    return fromEdgeList(leaves + 1, e, false);
}

CSR complete(uint32_t n) {
    EdgeList e;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1; j < n; ++j) e.push_back({i, j});
    return fromEdgeList(n, e, false);
}

static void addGrid(EdgeList& e, uint32_t base, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t v = base + r * cols + c;
            if (c + 1 < cols) e.push_back({v, v + 1});
            if (r + 1 < rows) e.push_back({v, v + cols});
        }
}

CSR twoComponents() {
    EdgeList e;
    for (uint32_t i = 0; i + 1 < 300; ++i) e.push_back({i, i + 1});
    addGrid(e, 300, 10, 10);
    return fromEdgeList(400, e, false);
}

CSR empty(uint32_t n) { return fromEdgeList(n, {}, false); }

CSR grid2d(uint32_t rows, uint32_t cols) {
    EdgeList e;
    addGrid(e, 0, rows, cols);
    return fromEdgeList((uint64_t)rows * cols, e, false);
}

CSR rmat(uint32_t scale, uint32_t edgefactor, bool directed, uint64_t seed) {
    const double a = 0.57, b = 0.19, c = 0.19;
    const uint64_t n = 1ull << scale, m = (uint64_t)edgefactor << scale;
    util::Rng rng(seed);
    EdgeList e;
    e.reserve(m);
    for (uint64_t i = 0; i < m; ++i) {
        // Recursively pick one of four quadrants of the adjacency matrix.
        uint32_t u = 0, v = 0;
        for (uint32_t bit = 0; bit < scale; ++bit) {
            double r = rng.uniform();
            uint32_t ub = 0, vb = 0;
            if (r < a) {}
            else if (r < a + b) vb = 1;
            else if (r < a + b + c) ub = 1;
            else { ub = 1; vb = 1; }
            u = (u << 1) | ub;
            v = (v << 1) | vb;
        }
        e.push_back({u, v});
    }
    // Self-loops and duplicates are removed by fromEdgeList; some vertices end
    // up isolated, which is realistic and exercises the "unreached" path.
    return fromEdgeList(n, e, directed);
}

std::vector<std::pair<std::string, CSR>> suite() {
    return {
        {"path1000", path(1000)},
        {"star100k", star(100000)},
        {"complete200", complete(200)},
        {"two_components", twoComponents()},
        {"empty1000", empty(1000)},
        {"grid100x100", grid2d(100, 100)},
        {"rmat12", rmat(12, 16, false, 42)},
        {"rmat12_directed", rmat(12, 16, true, 43)},
    };
}

} // namespace graph::gen
