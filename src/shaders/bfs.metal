// bfs.metal: GPU BFS kernels. Host side: src/metal/bfs_gpu.{hpp,mm}.
//
// Graph: symmetric CSR (row_ptr[n+1], col_idx[m]). dist[v] = BFS level, or
// INF_DIST if not reached yet. One dispatch per BFS level; the host reads the
// level statistics after each level to decide whether to continue.
//
// Level statistics (stats buffer, zeroed by the host before every level):
//   stats[0]  vertices in the input frontier
//   stats[1]  adjacency entries examined (sum of frontier degrees)
//   stats[2]  vertices discovered (next frontier size; queue tail)
//   stats[3]  sum of the discovered vertices' degrees (next level's edges)
//   stats[4..6]  gpu_binned: next-level small / medium / large bin sizes
// Counters are aggregated per simdgroup (simd_sum, then one atomic from lane
// 0): one atomic per 32 threads instead of one per thread, so a level with
// millions of frontier vertices does not serialize on a single address.

constant uint STAT_FRONTIER = 0;
constant uint STAT_EDGES = 1;
constant uint STAT_FOUND = 2;
constant uint STAT_NEXT_EDGES = 3;
constant uint STAT_BIN0 = 4;

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

// ===========================================================================
// Phase 5: load-balanced top-down BFS.
//
// Both variants keep ONE host sync per level (the ~111 us round trip
// measured in Phase 3) by doing all per-vertex bookkeeping at discovery
// time, in the previous level:
//   gpu_binned  a discovered vertex goes straight into the small / medium /
//               large queue for its degree, so the host learns the three bin
//               sizes from the same sync and encodes the three bin kernels
//               into one command buffer (no separate classification pass,
//               which would need a second sync to size its dispatches).
//   gpu_edge    a discovered vertex's degree is stored next to it
//               (next_deg[pos]); the next level scans those degrees in place
//               and the host gets the level's total edge count from stats[3].
//
// Queue appends are aggregated per simdgroup: one atomic per 32 lanes per
// round instead of one per discovered vertex (gpu_naive's second
// bottleneck). That requires every lane of the simdgroup to execute the
// append together, so all loops below that contain an append run a
// simdgroup-uniform number of rounds, with a per-lane `has` flag.

struct LevelParams {
    uint size;       // vertices in the input queue (of this bin)
    uint level;      // distance of the input frontier
    uint small_max;  // gpu_binned: degree <  small_max -> small bin
    uint large_min;  // gpu_binned: degree >= large_min -> large bin
    uint total;      // gpu_edge: edges to examine this level
    uint ept;        // gpu_edge: edges per thread
};

// Reserve slots for every lane with want == true; returns this lane's slot
// (meaningful only if want). All 32 lanes must call it together (lane 0
// does the atomic, simd_broadcast shares the base).
static inline uint simd_append(bool want, device atomic_uint* counter, uint lane)
{
    uint slot = simd_prefix_exclusive_sum(want ? 1u : 0u);
    uint total = simd_sum(want ? 1u : 0u);
    uint base = 0;
    if (lane == 0 && total) base = atomic_fetch_add_explicit(counter, total, memory_order_relaxed);
    return simd_broadcast(base, 0) + slot;
}

// Try to discover u at distance level+1. The relaxed load filters visited
// vertices cheaply before the CAS.
static inline bool try_discover(device atomic_uint* dist, uint u, uint level)
{
    return atomic_load_explicit(&dist[u], memory_order_relaxed) == INF_DIST &&
           claim(&dist[u], level + 1);
}

// Add the discovered degrees to the next level's edge count (one atomic per simdgroup).
static inline void add_next_edges(bool claimed, uint udeg, device atomic_uint* stats, uint lane)
{
    uint s = simd_sum(claimed ? udeg : 0u);
    if (lane == 0 && s) atomic_fetch_add_explicit(&stats[STAT_NEXT_EDGES], s, memory_order_relaxed);
}

// gpu_binned discovery: append u to the next-level queue of its degree bin.
static inline void bin_append(bool claimed, uint u, uint udeg, constant LevelParams& p,
                              device atomic_uint* stats, device uint* q_small,
                              device uint* q_medium, device uint* q_large, uint lane)
{
    // Most rounds discover nothing (only ~n of the ~m edges examined claim a
    // vertex), so skip the ~7 simd reductions below unless some lane claimed.
    // simd_any is uniform across the simdgroup, so convergence is preserved.
    if (!simd_any(claimed)) return;
    uint bin = udeg < p.small_max ? 0u : (udeg < p.large_min ? 1u : 2u);
    uint pos0 = simd_append(claimed && bin == 0, &stats[STAT_BIN0 + 0], lane);
    uint pos1 = simd_append(claimed && bin == 1, &stats[STAT_BIN0 + 1], lane);
    uint pos2 = simd_append(claimed && bin == 2, &stats[STAT_BIN0 + 2], lane);
    if (claimed) {
        if (bin == 0) q_small[pos0] = u;
        else if (bin == 1) q_medium[pos1] = u;
        else q_large[pos2] = u;
    }
    add_next_edges(claimed, udeg, stats, lane);
}

// Process neighbor e of the current vertex: shared body of the bin kernels.
#define BINNED_VISIT(has, e)                                                         \
    {                                                                                \
        uint u = (has) ? col_idx[e] : 0u;                                            \
        bool claimed = (has) && try_discover(dist, u, p.level);                      \
        uint udeg = claimed ? row_ptr[u + 1] - row_ptr[u] : 0u;                      \
        bin_append(claimed, u, udeg, p, stats, q_small, q_medium, q_large, lane);    \
    }

#define BINNED_ARGS                                                                  \
    device const uint* row_ptr  [[buffer(0)]],                                       \
    device const uint* col_idx  [[buffer(1)]],                                       \
    device atomic_uint* dist    [[buffer(2)]],                                       \
    device atomic_uint* stats   [[buffer(3)]],                                       \
    device const uint* queue    [[buffer(4)]],                                       \
    device uint* q_small        [[buffer(5)]],                                       \
    device uint* q_medium       [[buffer(6)]],                                       \
    device uint* q_large        [[buffer(7)]],                                       \
    constant LevelParams& p     [[buffer(8)]]

// Small bin (degree < small_max): one thread owns each vertex, but the
// simdgroup expands its 32 short lists *together* (warp-cooperative gather,
// Merrill et al.): the lists are concatenated via a prefix sum of degrees and
// in each round lane j takes edge j of the concatenation. The first version
// let every lane walk its own list (max-degree-in-simdgroup rounds): reads
// were uncoalesced and short-list lanes idled, and on LiveJournal it was
// 2x slower than putting every vertex in the medium bin (86.7 vs 42.3 ms).
kernel void bfs_binned_small(BINNED_ARGS,
                             uint gid  [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]])
{
    bool active = gid < p.size;  // no early return: all lanes must take part
    uint v = active ? queue[gid] : 0u;
    uint begin = active ? row_ptr[v] : 0u;
    uint deg = active ? row_ptr[v + 1] - begin : 0u;
    uint pre = simd_prefix_exclusive_sum(deg);  // start of this lane's list in the concatenation
    uint total = simd_sum(deg);
    for (uint base = 0; base < total; base += 32) {
        uint j = base + lane;
        // Owner of edge j = the largest lane l with pre[l] <= j (pre is
        // non-decreasing over lanes; zero-degree lanes are skipped by taking
        // the largest). Binary search over lanes with shuffles: 16+8+4+2+1.
        uint owner = 0;
        for (uint step = 16; step > 0; step >>= 1) {
            uint cand = owner + step;  // <= 31 always
            if (simd_shuffle(pre, cand) <= j) owner = cand;
        }
        uint ownerBegin = simd_shuffle(begin, owner);
        uint ownerPre = simd_shuffle(pre, owner);
        BINNED_VISIT(j < total, ownerBegin + (j - ownerPre))
    }
}

// Medium bin: simdgroup per vertex, lanes stride the adjacency list, so 32
// consecutive col_idx entries are read per round (coalesced). The early
// return is uniform for the whole simdgroup.
kernel void bfs_binned_medium(BINNED_ARGS,
                              uint gid  [[thread_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]])
{
    uint vi = gid / 32;
    if (vi >= p.size) return;
    uint v = queue[vi];
    uint begin = row_ptr[v], end = row_ptr[v + 1];
    for (uint base = begin; base < end; base += 32) BINNED_VISIT(base + lane < end, base + lane)
}

// Large bin: threadgroup per vertex, all threads stride the list. A single
// huge hub is still processed by ONE threadgroup (one GPU core); gpu_edge
// does not have that limit.
kernel void bfs_binned_large(BINNED_ARGS,
                             uint tgid [[threadgroup_position_in_grid]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint tgsz [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]])
{
    if (tgid >= p.size) return;  // uniform for the whole threadgroup
    uint v = queue[tgid];
    uint begin = row_ptr[v], end = row_ptr[v + 1];
    for (uint base = begin; base < end; base += tgsz) BINNED_VISIT(base + tid < end, base + tid)
}

// Largest i in [lo, hi] with off[i] <= e. With an exclusive scan of degrees
// in `off`, that is the frontier vertex owning edge e (zero-degree vertices
// share their successor's offset and are skipped because we take the largest).
static inline uint find_owner(device const uint* off, uint lo, uint hi, uint e)
{
    while (lo < hi) {
        uint mid = (lo + hi + 1) / 2;
        if (off[mid] <= e) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// gpu_edge: every thread handles `ept` edges of the level no matter how they
// are distributed over vertices, so a 100k-degree hub is spread over the
// whole GPU. Threadgroup t owns the edge chunk [t*tg*ept, (t+1)*tg*ept);
// within it thread j takes edges j, j+tg, j+2tg, ... so a simdgroup reads 32
// consecutive edges per round (coalesced col_idx). Owner lookup: thread 0
// binary-searches the chunk's first and last owner once; each edge then
// searches only that (usually tiny) range.
kernel void bfs_edge_expand(device const uint* row_ptr  [[buffer(0)]],
                            device const uint* col_idx  [[buffer(1)]],
                            device atomic_uint* dist    [[buffer(2)]],
                            device atomic_uint* stats   [[buffer(3)]],
                            device const uint* queue    [[buffer(4)]],
                            device const uint* offsets  [[buffer(5)]],
                            device uint* next_queue     [[buffer(6)]],
                            device uint* next_deg       [[buffer(7)]],
                            constant LevelParams& p     [[buffer(8)]],
                            uint tgid [[threadgroup_position_in_grid]],
                            uint tid  [[thread_index_in_threadgroup]],
                            uint tgsz [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]])
{
    threadgroup uint range[2];
    const uint chunk = tgsz * p.ept;
    const uint c0 = tgid * chunk;                    // < p.total (host sizes the grid)
    const uint c1 = min(c0 + chunk, p.total);
    if (tid == 0) {
        range[0] = find_owner(offsets, 0, p.size - 1, c0);
        range[1] = find_owner(offsets, range[0], p.size - 1, c1 - 1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lo = range[0], hi = range[1];

    for (uint k = 0; k < p.ept; ++k) {  // uniform round count: appends stay convergent
        uint e = c0 + k * tgsz + tid;
        bool has = e < c1;
        uint u = 0;
        if (has) {
            uint i = find_owner(offsets, lo, hi, e);
            uint v = queue[i];
            u = col_idx[row_ptr[v] + (e - offsets[i])];
        }
        bool claimed = has && try_discover(dist, u, p.level);
        if (!simd_any(claimed)) continue;  // nothing discovered this round (uniform branch)
        uint udeg = claimed ? row_ptr[u + 1] - row_ptr[u] : 0u;
        uint pos = simd_append(claimed, &stats[STAT_FOUND], lane);
        if (claimed) {
            next_queue[pos] = u;
            next_deg[pos] = udeg;  // becomes next level's scan input
        }
        add_next_edges(claimed, udeg, stats, lane);
    }
}

// ===========================================================================
// Phase 6: direction-optimizing BFS (gpu_do). Top-down levels reuse
// bfs_edge_expand; bottom-up levels use the kernel below on bitmap frontiers.

// deg[i] = degree(queue[i]): rebuilds gpu_edge's scan input when switching
// back to top-down (the queue then comes from bitmap_to_queue, which does not
// record degrees).
kernel void bfs_queue_degrees(device const uint* row_ptr [[buffer(0)]],
                              device const uint* queue   [[buffer(1)]],
                              device uint* deg           [[buffer(2)]],
                              constant uint& size        [[buffer(3)]],
                              uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    uint v = queue[gid];
    deg[gid] = row_ptr[v + 1] - row_ptr[v];
}

// Bottom-up step: thread per vertex. An unvisited vertex scans its neighbors
// and stops at the first one in the frontier bitmap (it only needs one
// parent). The random reads go to the bitmap (n/8 bytes: 600 KB for
// LiveJournal, cache-resident) instead of the 19 MB dist array.
// No CAS is needed: only vertex v's own thread writes dist[v].
//
// Next frontier: the 32 lanes of a simdgroup are 32 consecutive vertices
// gid & ~31 .. +31 (1D threadgroups, multiples of 32), which is exactly one
// bitmap word. simd_or assembles the word and lane 0 stores it: no atomics,
// and every word is overwritten, so `next` needs no clearing.
// Stats: STAT_FOUND = vertices found (next frontier size), STAT_EDGES =
// adjacency entries examined, STAT_NEXT_EDGES = their degrees (the next
// top-down level's edge count, if the heuristic switches back).
kernel void bfs_bottom_up(device const uint* row_ptr [[buffer(0)]],
                          device const uint* col_idx [[buffer(1)]],
                          device uint* dist          [[buffer(2)]],
                          device const uint* front   [[buffer(3)]],
                          device uint* next          [[buffer(4)]],
                          device atomic_uint* stats  [[buffer(5)]],
                          constant uint& n           [[buffer(6)]],
                          constant uint& level       [[buffer(7)]],
                          uint gid  [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]])
{
    bool found = false;
    uint examined = 0, ndeg = 0;
    if (gid < n && dist[gid] == INF_DIST) {
        uint begin = row_ptr[gid], end = row_ptr[gid + 1];
        for (uint e = begin; e < end; ++e) {
            ++examined;
            uint u = col_idx[e];
            if ((front[u >> 5] >> (u & 31)) & 1u) { found = true; break; }  // early exit
        }
        if (found) {
            dist[gid] = level + 1;
            ndeg = end - begin;
        }
    }
    // No early return above: every lane reaches the simdgroup operations.
    uint word = simd_or(found ? (1u << lane) : 0u);
    if (lane == 0 && gid < n) next[gid >> 5] = word;
    uint f = simd_sum(found ? 1u : 0u), x = simd_sum(examined), d = simd_sum(ndeg);
    if (lane == 0) {
        if (f) atomic_fetch_add_explicit(&stats[STAT_FOUND], f, memory_order_relaxed);
        if (x) atomic_fetch_add_explicit(&stats[STAT_EDGES], x, memory_order_relaxed);
        if (d) atomic_fetch_add_explicit(&stats[STAT_NEXT_EDGES], d, memory_order_relaxed);
    }
}
