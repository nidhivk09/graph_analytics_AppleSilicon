// primitives.metal: generic data-parallel building blocks reused by the BFS
// and PageRank kernels. Host wrappers: src/metal/primitives.{hpp,mm}.
// Every kernel takes its element count and bounds-checks (grids are rounded
// up to whole threadgroups).

// buf[i] = value for i < n.
kernel void fill_u32(device uint* buf     [[buffer(0)]],
                     constant uint& value [[buffer(1)]],
                     constant uint& n     [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid < n) buf[gid] = value;
}

// ---------------------------------------------------------------------------
// Reduction, two passes with the same kernel: pass 1 runs G threadgroups over
// the input (grid-stride) and writes one partial per threadgroup; pass 2 runs
// one threadgroup over the G partials and writes the final value to out[0].
// Two fixed passes instead of atomics: deterministic and contention-free.
// Sums wrap modulo 2^32 (callers keep totals below that, e.g. edge counts).

kernel void reduce_sum_u32(device const uint* in [[buffer(0)]],
                           device uint* out      [[buffer(1)]],
                           constant uint& n      [[buffer(2)]],
                           uint gid  [[thread_position_in_grid]],
                           uint grid [[threads_per_grid]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]],
                           uint sid  [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup uint scratch[32];
    uint acc = 0;
    for (uint i = gid; i < n; i += grid) acc += in[i];
    acc = tg_sum(acc, scratch, sid, lane, nsg);
    if (tid == 0) out[tgid] = acc;
}

kernel void reduce_max_u32(device const uint* in [[buffer(0)]],
                           device uint* out      [[buffer(1)]],
                           constant uint& n      [[buffer(2)]],
                           uint gid  [[thread_position_in_grid]],
                           uint grid [[threads_per_grid]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]],
                           uint sid  [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup uint scratch[32];
    uint acc = 0;
    for (uint i = gid; i < n; i += grid) acc = max(acc, in[i]);
    acc = tg_max(acc, scratch, sid, lane, nsg);
    if (tid == 0) out[tgid] = acc;
}

// ---------------------------------------------------------------------------
// Exclusive scan of uint32, three phases (the host recurses on phase 2):
//   1. scan_block: each threadgroup scans SCAN_BLOCK consecutive elements and
//      writes its block total to block_sums[tgid].
//   2. the block sums are scanned (recursively, same kernels).
//   3. scan_add: add the scanned block sum to every element of the block.
// In place (in == out) is allowed: each block loads everything before it
// stores, and blocks are disjoint.
//
// Layout: 256 threads x 4 consecutive elements per thread = 1024 per block.
// Per thread: serial scan of its 4 values; per simdgroup: simd_prefix_
// exclusive_sum of the thread totals; per threadgroup: scan of the 8
// simdgroup totals by the first simdgroup.
constant uint SCAN_TG = 256;
constant uint SCAN_ITEMS = 4;
constant uint SCAN_BLOCK = SCAN_TG * SCAN_ITEMS;

kernel void scan_block(device const uint* in   [[buffer(0)]],
                       device uint* out        [[buffer(1)]],
                       device uint* block_sums [[buffer(2)]],
                       constant uint& n        [[buffer(3)]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint tid  [[thread_index_in_threadgroup]],
                       uint sid  [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]])
{
    threadgroup uint sg_off[SCAN_TG / 32];
    threadgroup uint block_total;

    const uint base = tgid * SCAN_BLOCK + tid * SCAN_ITEMS;
    uint v[SCAN_ITEMS];
    uint thread_total = 0;
    for (uint k = 0; k < SCAN_ITEMS; ++k) {
        v[k] = base + k < n ? in[base + k] : 0u;
        thread_total += v[k];
    }

    uint pre = simd_prefix_exclusive_sum(thread_total);
    if (lane == 31) sg_off[sid] = pre + thread_total;  // simdgroup total
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        uint s = lane < SCAN_TG / 32 ? sg_off[lane] : 0u;
        uint s_pre = simd_prefix_exclusive_sum(s);
        if (lane < SCAN_TG / 32) sg_off[lane] = s_pre;
        if (lane == SCAN_TG / 32 - 1) block_total = s_pre + s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint run = sg_off[sid] + pre;
    for (uint k = 0; k < SCAN_ITEMS; ++k) {
        if (base + k < n) out[base + k] = run;
        run += v[k];
    }
    if (tid == 0) block_sums[tgid] = block_total;
}

kernel void scan_add(device uint* out              [[buffer(0)]],
                     device const uint* block_sums [[buffer(1)]],
                     constant uint& n              [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid < n) out[gid] += block_sums[gid / SCAN_BLOCK];
}

// ---------------------------------------------------------------------------
// Frontier format conversion (direction-optimizing BFS switches between a
// vertex queue for top-down and a bitmap for bottom-up).
// Bitmap: bit (v % 32) of word v / 32. The caller zeroes it first (fill_u32).

kernel void queue_to_bitmap(device const uint* queue  [[buffer(0)]],
                            device atomic_uint* bitmap [[buffer(1)]],
                            constant uint& size        [[buffer(2)]],
                            uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    uint v = queue[gid];
    atomic_fetch_or_explicit(&bitmap[v >> 5], 1u << (v & 31), memory_order_relaxed);
}

// Thread per vertex; appends set vertices to `queue` and counts them in
// size[0] (caller zeroes it). One atomic per simdgroup: a ballot gives each
// lane its slot among the set lanes, lane 0 reserves the whole range.
// Queue order is not deterministic (simdgroups race for ranges).
kernel void bitmap_to_queue(device const uint* bitmap [[buffer(0)]],
                            device uint* queue        [[buffer(1)]],
                            device atomic_uint* size  [[buffer(2)]],
                            constant uint& n          [[buffer(3)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]])
{
    bool set = gid < n && (bitmap[gid >> 5] >> (gid & 31) & 1u);
    uint count = simd_sum(set ? 1u : 0u);
    uint slot = simd_prefix_exclusive_sum(set ? 1u : 0u);
    uint start = 0;
    if (lane == 0 && count > 0) start = atomic_fetch_add_explicit(size, count, memory_order_relaxed);
    start = simd_broadcast(start, 0);
    if (set) queue[start + slot] = gid;
}
