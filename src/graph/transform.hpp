#pragma once
// Derived graphs, built in memory at load time and never stored (PLAN.md 4.4).
#include "graph/csr.hpp"

namespace graph {

// Reverse every edge. For PageRank pull: transpose(G) lists in-neighbors.
// Output lists come out sorted because sources are scattered in increasing order.
CSR transpose(const CSR& g);

// Union of G and transpose(G), sorted and deduplicated, flagged undirected.
// Undirected inputs are returned as-is (shared storage, no copy).
CSR symmetrize(const CSR& g);

} // namespace graph
