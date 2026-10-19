#pragma once
// Single BFS level steps on the CPU, operating on the SAME arrays the GPU
// kernels use (Metal shared buffers: unified memory, no copies). The hybrid
// BFS (Phase 7) runs each level on either device, so the CPU steps keep the
// GPU's data layout exactly:
//   top-down:  queue in -> queue out + each discovered vertex's degree next
//              to it (gpu_edge's scan input), dist claimed by CAS
//   bottom-up: bitmap in -> bitmap out (whole words written, no clearing)
// and return the same counters the GPU kernels write into their stats buffer.
#include "graph/csr.hpp"

#include <cstdint>

namespace cpu::steps {

struct StepOut {
    uint64_t found = 0;      // vertices discovered (next frontier size)
    uint64_t examined = 0;   // adjacency entries examined
    uint64_t nextEdges = 0;  // sum of discovered vertices' degrees
};

// Levels with fewer edges than this run on one thread: an OpenMP parallel
// region costs ~20 us across both clusters (Phase 2), more than the whole
// level. The hybrid's CPU levels are small by design, so this matters.
constexpr uint64_t kSerialEdges = 2048;

// dist[v] = INF except dist[source] = 0.
void initDist(uint32_t* dist, uint64_t n, uint32_t source, int threads);

// Top-down level: expand q[0..size) (distance `level`); `edges` = sum of
// their degrees (picks serial vs parallel).
StepOut topDown(const graph::CSR& g, uint32_t* dist, const uint32_t* q, uint64_t size,
                uint64_t edges, uint32_t* qNext, uint32_t* degNext, uint32_t level, int threads);

// Bottom-up level on bitmaps (bit v%32 of word v/32): every unvisited vertex
// looks for a neighbor in `front`, stopping at the first.
StepOut bottomUp(const graph::CSR& g, uint32_t* dist, const uint32_t* front, uint32_t* next,
                 uint32_t level, int threads);

// Clear `bitmap` (ceil(n/32) words) and set the bit of every queue entry.
void queueToBitmap(const uint32_t* q, uint64_t size, uint32_t* bitmap, uint64_t n, int threads);

// Bitmap -> queue in increasing vertex order, with degrees; returns the count.
uint64_t bitmapToQueue(const graph::CSR& g, const uint32_t* bitmap, uint32_t* q, uint32_t* deg,
                       int threads);

} // namespace cpu::steps
