#pragma once
// Vertex orderings (PLAN.md 5.3). Each returns a permutation perm with
// perm[old] = new; transform.hpp's permute() relabels a CSR with it.
// Orderings are computed on the symmetrized graph (undirected structure);
// PageRank then permutes the directed graph with the same perm.
#include "graph/csr.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace graph::reorder {

// original (identity), random (fixed seed), degree (descending degree,
// ties by original id), rcm (Reverse Cuthill-McKee, per component).
std::vector<uint32_t> compute(const CSR& sym, const std::string& ordering, uint64_t seed = 1);

// True if p is a permutation of 0..n-1.
bool isPermutation(const std::vector<uint32_t>& p);

// Locality of the current labeling, over all stored adjacency entries:
// mean |u - v|, and the fraction with u / 64 == v / 64 (both endpoints in
// the same 64-vertex block: the same 256-byte stretch of a 4-byte array).
struct Locality { double meanGap = 0, sameBlock64 = 0; };
Locality locality(const CSR& g);

} // namespace graph::reorder
