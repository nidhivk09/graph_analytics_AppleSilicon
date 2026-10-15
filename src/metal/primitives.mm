#include "metal/primitives.hpp"

#include <algorithm>
#include <stdexcept>

namespace gpu {

namespace {
constexpr size_t kTG = 256;
constexpr size_t kMaxReduceGroups = 1024;  // pass-2 threadgroup handles <= 1024 partials

void setU32(id<MTLComputeCommandEncoder> enc, uint32_t v, NSUInteger index) {
    [enc setBytes:&v length:sizeof v atIndex:index];
}
} // namespace

Primitives::Primitives(MetalContext& ctx) : ctx_(ctx) {
    fill_ = ctx.pipeline("primitives", "fill_u32");
    sum_ = ctx.pipeline("primitives", "reduce_sum_u32");
    max_ = ctx.pipeline("primitives", "reduce_max_u32");
    scanBlock_ = ctx.pipeline("primitives", "scan_block");
    scanAdd_ = ctx.pipeline("primitives", "scan_add");
    q2b_ = ctx.pipeline("primitives", "queue_to_bitmap");
    b2q_ = ctx.pipeline("primitives", "bitmap_to_queue");
}

id<MTLBuffer> Primitives::scratch(std::vector<id<MTLBuffer>>& pool, size_t i, size_t bytes) {
    if (pool.size() <= i) pool.resize(i + 1, nil);
    if (!pool[i] || pool[i].length < bytes) pool[i] = ctx_.buffer(std::max<size_t>(bytes, 4096));
    return pool[i];
}

void Primitives::fill(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf, uint32_t value, size_t n,
                      size_t offsetBytes) {
    if (n == 0) return;
    [enc setBuffer:buf offset:offsetBytes atIndex:0];
    setU32(enc, value, 1);
    setU32(enc, (uint32_t)n, 2);
    MetalContext::dispatch1D(enc, fill_, n, kTG);
}

void Primitives::reduce(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                        id<MTLBuffer> in, size_t n, id<MTLBuffer> out) {
    // Pass 1: G groups, grid-stride over n. Pass 2: one group of 1024 over G.
    size_t groups = std::min(kMaxReduceGroups, std::max<size_t>(1, (n + kTG - 1) / kTG));
    id<MTLBuffer> part = scratch(partials_, 0, groups * 4);
    [enc setComputePipelineState:pso];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:part offset:0 atIndex:1];
    setU32(enc, (uint32_t)n, 2);
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];

    [enc setBuffer:part offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    setU32(enc, (uint32_t)groups, 2);
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

void Primitives::reduceSum(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, size_t n,
                           id<MTLBuffer> out) {
    reduce(enc, sum_, in, n, out);
}

void Primitives::reduceMax(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, size_t n,
                           id<MTLBuffer> out) {
    reduce(enc, max_, in, n, out);
}

void Primitives::scanExclusive(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in,
                               id<MTLBuffer> out, size_t n, id<MTLBuffer> total) {
    if (n == 0) {
        if (total) fill(enc, total, 0, 1);
        return;
    }
    scanLevel(enc, in, out, n, total, 0);
}

void Primitives::scanLevel(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, id<MTLBuffer> out,
                           size_t n, id<MTLBuffer> total, size_t depth) {
    const size_t blocks = (n + kScanBlock - 1) / kScanBlock;
    // A single block's total is the grand total: write it straight to `total`.
    id<MTLBuffer> sums = blocks == 1 ? total : scratch(scanSums_, depth, blocks * 4);
    if (!sums) sums = scratch(scanSums_, depth, 4);  // single block, caller wants no total

    [enc setComputePipelineState:scanBlock_];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBuffer:sums offset:0 atIndex:2];
    setU32(enc, (uint32_t)n, 3);
    [enc dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kScanBlock / 4, 1, 1)];
    if (blocks == 1) return;

    // Scan the block sums in place (recursion depth is log_1024 n: 3 levels
    // cover 2^30 elements), then add them back.
    scanLevel(enc, sums, sums, blocks, total, depth + 1);
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:sums offset:0 atIndex:1];
    setU32(enc, (uint32_t)n, 2);
    MetalContext::dispatch1D(enc, scanAdd_, n, kTG);
}

void Primitives::queueToBitmap(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> queue, size_t size,
                               id<MTLBuffer> bitmap, size_t n) {
    fill(enc, bitmap, 0, (n + 31) / 32);
    if (size == 0) return;
    [enc setBuffer:queue offset:0 atIndex:0];
    [enc setBuffer:bitmap offset:0 atIndex:1];
    setU32(enc, (uint32_t)size, 2);
    MetalContext::dispatch1D(enc, q2b_, size, kTG);
}

void Primitives::bitmapToQueue(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> bitmap, size_t n,
                               id<MTLBuffer> queue, id<MTLBuffer> size) {
    fill(enc, size, 0, 1);
    if (n == 0) return;
    [enc setBuffer:bitmap offset:0 atIndex:0];
    [enc setBuffer:queue offset:0 atIndex:1];
    [enc setBuffer:size offset:0 atIndex:2];
    setU32(enc, (uint32_t)n, 3);
    MetalContext::dispatch1D(enc, b2q_, n, kTG);
}

template <class F>
void Primitives::runSync(F encode) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode(enc);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error("primitives: command buffer failed");
    }
}

uint32_t Primitives::runReduceSum(const std::vector<uint32_t>& v) {
    id<MTLBuffer> in = ctx_.buffer(v.data(), v.size() * 4), out = ctx_.buffer(4);
    runSync([&](id<MTLComputeCommandEncoder> enc) { reduceSum(enc, in, v.size(), out); });
    return *static_cast<uint32_t*>(out.contents);
}

uint32_t Primitives::runReduceMax(const std::vector<uint32_t>& v) {
    id<MTLBuffer> in = ctx_.buffer(v.data(), v.size() * 4), out = ctx_.buffer(4);
    runSync([&](id<MTLComputeCommandEncoder> enc) { reduceMax(enc, in, v.size(), out); });
    return *static_cast<uint32_t*>(out.contents);
}

std::vector<uint32_t> Primitives::runScan(const std::vector<uint32_t>& v, uint32_t* total) {
    id<MTLBuffer> in = ctx_.buffer(v.data(), v.size() * 4), out = ctx_.buffer(v.size() * 4);
    id<MTLBuffer> tot = ctx_.buffer(4);
    *static_cast<uint32_t*>(tot.contents) = 0xDEADBEEF;  // must be overwritten
    runSync([&](id<MTLComputeCommandEncoder> enc) { scanExclusive(enc, in, out, v.size(), tot); });
    if (total) *total = *static_cast<uint32_t*>(tot.contents);
    const uint32_t* o = static_cast<const uint32_t*>(out.contents);
    return std::vector<uint32_t>(o, o + v.size());
}

} // namespace gpu
