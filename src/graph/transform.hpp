#pragma once
// Derived graphs, built in memory at load time and never stored (PLAN.md 4.4).
#include "graph/csr.hpp"

#include <vector>

namespace graph {

// Reverse every edge. For PageRank pull: transpose(G) lists in-neighbors.
// Output lists come out sorted because sources are scattered in increasing order.
CSR transpose(const CSR& g);

// Union of G and transpose(G), sorted and deduplicated, flagged undirected.
// Undirected inputs are returned as-is (shared storage, no copy).
CSR symmetrize(const CSR& g);

// Relabel vertices: vertex v becomes perm[v] (perm a permutation of 0..n-1).
// Adjacency lists are re-sorted, so the sorted/dedup flags stay valid.
CSR permute(const CSR& g, const std::vector<uint32_t>& perm);

} // namespace graph
