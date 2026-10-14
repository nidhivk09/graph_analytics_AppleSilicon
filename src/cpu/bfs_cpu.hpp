#pragma once
// CPU BFS. cpu_serial is the reference every other variant is checked against.
#include "graph/csr.hpp"

#include <vector>

namespace cpu {

constexpr uint32_t kUnreached = 0xFFFFFFFFu;

// Queue-based serial BFS; returns dist[v] (kUnreached if not reachable).
std::vector<uint32_t> bfsSerial(const graph::CSR& g, uint32_t source);

} // namespace cpu
