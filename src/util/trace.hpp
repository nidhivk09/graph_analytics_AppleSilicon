#pragma once
// Per-level BFS trace record (--trace). Shared by CPU, GPU and hybrid BFS.
#include <cstdint>

namespace util {

struct BfsLevel {
    uint32_t level;          // distance being assigned to newly found vertices - 1
    uint64_t frontier;       // vertices in the input frontier of this level
    uint64_t edges;          // adjacency entries examined during the level
    char direction;          // 'T' = top-down, 'B' = bottom-up
    char device;             // 'C' = CPU, 'G' = GPU
    double time_ms;          // wall-clock time of the level
};

} // namespace util
