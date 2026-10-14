#pragma once
// Synthetic test graphs (PLAN.md 4.5): the core unit-test inputs. Each one
// stresses something specific (depth, imbalance, density, disconnection).
#include "graph/csr.hpp"

#include <string>
#include <utility>
#include <vector>

namespace graph::gen {

CSR path(uint32_t n);                         // 0-1-2-...: depth n-1, frontier size 1
CSR star(uint32_t leaves);                    // hub 0 + leaves: extreme degree imbalance
CSR complete(uint32_t n);                     // every pair connected: depth 1, dense
CSR twoComponents();                          // path(300) + grid(10x10), disconnected
CSR empty(uint32_t n);                        // n isolated vertices, no edges
CSR grid2d(uint32_t rows, uint32_t cols);     // 4-neighbour mesh: road-like, high diameter
// R-MAT / Kronecker with Graph500 parameters a=0.57, b=0.19, c=0.19 (d=0.05):
// 2^scale vertices, edgefactor*2^scale generated edges (before dedup).
CSR rmat(uint32_t scale, uint32_t edgefactor, bool directed, uint64_t seed);

// The whole suite, as used by the unit tests. Includes a directed R-MAT
// (for PageRank / transpose tests) alongside the undirected graphs.
std::vector<std::pair<std::string, CSR>> suite();

} // namespace graph::gen
