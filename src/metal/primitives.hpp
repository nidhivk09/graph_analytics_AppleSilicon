#pragma once
// Host wrappers for src/shaders/primitives.metal.
//
// Every operation *encodes* into a caller-provided compute encoder and does
// not commit, so BFS/PageRank can chain primitives with their own kernels in
// one command buffer. Dispatches in one (serial) encoder run in order with
// memory visible between them, so results can feed the next dispatch.
// Results that the host needs (a sum, a scan total) land in a buffer the
// caller reads after waitUntilCompleted.
//
// The run*() helpers do commit + wait; they exist for tests and one-off use.
#include "metal/context.hpp"

#include <vector>

namespace gpu {

class Primitives {
public:
    explicit Primitives(MetalContext& ctx);

    // buf[offset/4 + i] = value for i < n.
    void fill(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf, uint32_t value, size_t n,
              size_t offsetBytes = 0);

    // out[0] = sum (mod 2^32) / max of in[0..n). n = 0 gives 0.
    void reduceSum(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, size_t n, id<MTLBuffer> out);
    void reduceMax(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, size_t n, id<MTLBuffer> out);

    // out[i] = in[0] + ... + in[i-1]; in == out is allowed. If `total` is not
    // nil, total[0] = sum of all n inputs (the next frontier size, typically).
    void scanExclusive(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, id<MTLBuffer> out,
                       size_t n, id<MTLBuffer> total = nil);

    // Zero `bitmap` (ceil(n/32) words), then set the bit of each queue entry.
    void queueToBitmap(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> queue, size_t size,
                       id<MTLBuffer> bitmap, size_t n);

    // Append every set vertex v < n to `queue` (unordered); size[0] = count.
    void bitmapToQueue(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> bitmap, size_t n,
                       id<MTLBuffer> queue, id<MTLBuffer> size);

    // Synchronous helpers on host vectors (tests).
    uint32_t runReduceSum(const std::vector<uint32_t>& v);
    uint32_t runReduceMax(const std::vector<uint32_t>& v);
    std::vector<uint32_t> runScan(const std::vector<uint32_t>& v, uint32_t* total);

    static constexpr size_t kScanBlock = 1024;  // must match SCAN_BLOCK in primitives.metal

private:
    void reduce(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                id<MTLBuffer> in, size_t n, id<MTLBuffer> out);
    void scanLevel(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, id<MTLBuffer> out, size_t n,
                   id<MTLBuffer> total, size_t depth);
    id<MTLBuffer> scratch(std::vector<id<MTLBuffer>>& pool, size_t i, size_t bytes);
    template <class F> void runSync(F encode);

    MetalContext& ctx_;
    id<MTLComputePipelineState> fill_, sum_, max_, scanBlock_, scanAdd_, q2b_, b2q_;
    // Scratch reused across calls (allocation is far too slow for timed loops).
    // Reuse within one command buffer is safe: the serial encoder orders all
    // dispatches that touch a buffer.
    std::vector<id<MTLBuffer>> scanSums_;  // one per recursion depth
    std::vector<id<MTLBuffer>> partials_;  // [0]: reduction partials
};

} // namespace gpu
