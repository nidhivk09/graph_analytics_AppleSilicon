// common.h: shared by every kernel file.
// The host prepends this file to each .metal source before runtime compilation,
// because newLibraryWithSource does not resolve local #includes.
#include <metal_stdlib>
using namespace metal;

// Marks an unreached vertex in BFS distance arrays (matches the host constant).
constant uint INF_DIST = 0xFFFFFFFFu;

// Threadgroup-wide reductions. Every thread of the threadgroup must call them
// (they contain barriers); the result is returned to all threads. `scratch`
// is >= 32 uints of threadgroup memory declared by the calling kernel (MSL
// only allows threadgroup variables at kernel scope). The trailing barrier
// makes it safe to reuse `scratch` right after the call.
static inline uint tg_sum(uint v, threadgroup uint* scratch, uint sid, uint lane, uint nsg)
{
    v = simd_sum(v);
    if (lane == 0) scratch[sid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint w = lane < nsg ? scratch[lane] : 0u;
    w = simd_sum(w);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return w;
}

static inline uint tg_max(uint v, threadgroup uint* scratch, uint sid, uint lane, uint nsg)
{
    v = simd_max(v);
    if (lane == 0) scratch[sid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint w = lane < nsg ? scratch[lane] : 0u;
    w = simd_max(w);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return w;
}
