// micro.metal: microbenchmark and smoke-test kernels.

// Phase 0 hello world: c = a + b. The explicit count + bounds check is the
// project-wide rule: the grid may be rounded up past the end of the buffers.
kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float*       c [[buffer(2)]],
                       constant uint&      n [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    c[gid] = a[gid] + b[gid];
}

// ---------------------------------------------------------------------------
// Phase 3 microbenchmarks.
//
// All bandwidth kernels use a fixed-size grid with a grid-stride loop: thread
// g touches elements g, g+G, g+2G, ... so a simdgroup always covers 32
// consecutive elements (fully coalesced), and the grid size is a knob that is
// independent of the buffer size.
//
// Data is uint, not float: the read/gather sums wrap modulo 2^32, so the CPU
// can verify them exactly regardless of summation order.

// Reduce one uint per thread to one uint per threadgroup and store it in
// partial[tgid]. One store per threadgroup (a few KB per dispatch) keeps the
// result live, so the compiler cannot drop the loads, without adding
// measurable write traffic even at 1 MB buffer sizes. `sg` is 32 uints of
// threadgroup scratch, declared in the kernel (MSL only allows threadgroup
// variables at kernel scope).
static void tg_reduce_store(uint v, device uint* partial, threadgroup uint* sg,
                            uint tgid, uint sid, uint lane, uint nsg)
{
    v = simd_sum(v);
    if (lane == 0) sg[sid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        uint w = lane < nsg ? sg[lane] : 0u;
        w = simd_sum(w);
        if (lane == 0) partial[tgid] = w;
    }
}

// Sequential read: sum n4 uint4 elements (16-byte loads, the widest natural
// vector load).
kernel void bw_read(device const uint4* a   [[buffer(0)]],
                    device uint* partial    [[buffer(1)]],
                    constant uint& n4       [[buffer(2)]],
                    uint gid  [[thread_position_in_grid]],
                    uint grid [[threads_per_grid]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint sid  [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup uint sg[32];
    uint4 acc = 0;
    for (uint i = gid; i < n4; i += grid) acc += a[i];
    // Every thread must reach the barrier inside, including those with no work.
    tg_reduce_store(acc.x + acc.y + acc.z + acc.w, partial, sg, tgid, sid, lane, nsg);
}

// Sequential write: a[i] = (i, i, i, i). A value that depends on i makes the
// result checkable and rules out any "fill with constant" shortcut.
kernel void bw_write(device uint4* a      [[buffer(0)]],
                     constant uint& n4    [[buffer(1)]],
                     uint gid  [[thread_position_in_grid]],
                     uint grid [[threads_per_grid]])
{
    for (uint i = gid; i < n4; i += grid) a[i] = uint4(i);
}

// Sequential copy: b[i] = a[i].
kernel void bw_copy(device const uint4* a [[buffer(0)]],
                    device uint4* b       [[buffer(1)]],
                    constant uint& n4     [[buffer(2)]],
                    uint gid  [[thread_position_in_grid]],
                    uint grid [[threads_per_grid]])
{
    for (uint i = gid; i < n4; i += grid) b[i] = a[i];
}

// Gather: sum table[idx[i]] for i < m. idx is streamed sequentially and the
// table is read through it, which is exactly the CSR access pattern (stream
// col_idx, gather per-vertex data). With idx[i] = i this is a sequential read.
kernel void gather_sum(device const uint* table [[buffer(0)]],
                       device const uint* idx   [[buffer(1)]],
                       device uint* partial     [[buffer(2)]],
                       constant uint& m         [[buffer(3)]],
                       uint gid  [[thread_position_in_grid]],
                       uint grid [[threads_per_grid]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint sid  [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup uint sg[32];
    uint acc = 0;
    for (uint i = gid; i < m; i += grid) acc += table[idx[i]];
    tg_reduce_store(acc, partial, sg, tgid, sid, lane, nsg);
}

// Does nothing: measures pure dispatch / command-buffer overhead.
kernel void empty_kernel() {}
