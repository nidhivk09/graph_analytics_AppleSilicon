// bfs.metal: GPU BFS kernels. Host side: src/metal/bfs_gpu.{hpp,mm}.
//
// Graph: symmetric CSR (row_ptr[n+1], col_idx[m]). dist[v] = BFS level, or
// INF_DIST if not reached yet. One dispatch per BFS level; the host reads the
// level statistics after each level to decide whether to continue.
//
// Level statistics (stats buffer, zeroed by the host before every level):
//   stats[0]  vertices in the input frontier
//   stats[1]  adjacency entries examined (sum of frontier degrees)
//   stats[2]  vertices discovered (next frontier size)
// Counters are aggregated per simdgroup (simd_sum, then one atomic from lane
// 0): one atomic per 32 threads instead of one per thread, so a level with
// millions of frontier vertices does not serialize on a single address.

constant uint STAT_FRONTIER = 0;
constant uint STAT_EDGES = 1;
constant uint STAT_FOUND = 2;

// dist[v] = (v == source) ? 0 : INF_DIST.
kernel void bfs_init(device uint* dist      [[buffer(0)]],
                     constant uint& n       [[buffer(1)]],
                     constant uint& source  [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid < n) dist[gid] = gid == source ? 0u : INF_DIST;
}

static inline void add_stats(device atomic_uint* stats, uint frontier, uint edges, uint found,
                             uint lane)
{
    frontier = simd_sum(frontier);
    edges = simd_sum(edges);
    found = simd_sum(found);
    if (lane == 0) {
        if (frontier) atomic_fetch_add_explicit(&stats[STAT_FRONTIER], frontier, memory_order_relaxed);
        if (edges) atomic_fetch_add_explicit(&stats[STAT_EDGES], edges, memory_order_relaxed);
        if (found) atomic_fetch_add_explicit(&stats[STAT_FOUND], found, memory_order_relaxed);
    }
}

// gpu_vertex: one thread per vertex, every level. Threads whose vertex is not
// on the frontier (dist != level) do nothing: simple, no queue, but the whole
// vertex range is launched and dist[] is scanned every level.
//
// Plain (non-atomic) dist accesses on purpose: every writer stores the same
// value (level + 1) and readers only test for INF_DIST, so a racing thread at
// worst re-writes the same value. The cost is that STAT_FOUND can overcount
// when two frontier vertices discover the same neighbor in the same level; it
// is used only as a "did anything change" flag (exact counts come from
// stats[0] of the next level).
kernel void bfs_vertex_level(device const uint* row_ptr [[buffer(0)]],
                             device const uint* col_idx [[buffer(1)]],
                             device uint* dist          [[buffer(2)]],
                             device atomic_uint* stats  [[buffer(3)]],
                             constant uint& n           [[buffer(4)]],
                             constant uint& level       [[buffer(5)]],
                             uint gid  [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]])
{
    // No early return: every lane must reach the simd_sum in add_stats.
    uint frontier = 0, edges = 0, found = 0;
    if (gid < n && dist[gid] == level) {
        uint begin = row_ptr[gid], end = row_ptr[gid + 1];
        frontier = 1;
        edges = end - begin;
        for (uint e = begin; e < end; ++e) {
            uint u = col_idx[e];
            if (dist[u] == INF_DIST) {
                dist[u] = level + 1;
                found++;
            }
        }
    }
    add_stats(stats, frontier, edges, found, lane);
}

// Atomically set *d from INF_DIST to `value`. True for exactly one caller.
// MSL only has the *weak* compare-exchange, which may fail spuriously (with
// *d still INF_DIST); without the retry such a vertex would never be claimed.
static inline bool claim(device atomic_uint* d, uint value)
{
    uint expected = INF_DIST;
    while (!atomic_compare_exchange_weak_explicit(d, &expected, value, memory_order_relaxed,
                                                  memory_order_relaxed)) {
        if (expected != INF_DIST) return false;  // someone else claimed it
        expected = INF_DIST;                     // spurious failure: retry
    }
    return true;
}

// gpu_naive: one thread per frontier vertex (queue in, queue out). A neighbor
// is claimed with compare-and-swap on dist, so exactly one thread appends it
// to the next frontier, which keeps the queue duplicate-free and STAT_FOUND
// exact. The plain atomic load first skips the CAS for already-visited
// neighbors (the common case on social graphs after the first few levels).
// Each successful claim does its own atomic_fetch_add on the queue tail:
// this is the "naive" part, along with one thread walking a whole adjacency
// list no matter how long (load imbalance on hubs).
kernel void bfs_naive_level(device const uint* row_ptr [[buffer(0)]],
                            device const uint* col_idx [[buffer(1)]],
                            device atomic_uint* dist   [[buffer(2)]],
                            device atomic_uint* stats  [[buffer(3)]],
                            device const uint* queue   [[buffer(4)]],
                            device uint* next_queue    [[buffer(5)]],
                            constant uint& size        [[buffer(6)]],
                            constant uint& level       [[buffer(7)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]])
{
    uint frontier = 0, edges = 0;
    if (gid < size) {
        uint v = queue[gid];
        uint begin = row_ptr[v], end = row_ptr[v + 1];
        frontier = 1;
        edges = end - begin;
        for (uint e = begin; e < end; ++e) {
            uint u = col_idx[e];
            if (atomic_load_explicit(&dist[u], memory_order_relaxed) != INF_DIST) continue;
            if (claim(&dist[u], level + 1)) {
                // Queue position = the global discovered count before this claim.
                uint pos = atomic_fetch_add_explicit(&stats[STAT_FOUND], 1u, memory_order_relaxed);
                next_queue[pos] = u;
            }
        }
    }
    add_stats(stats, frontier, edges, 0u, lane);  // STAT_FOUND already counted per claim
}
