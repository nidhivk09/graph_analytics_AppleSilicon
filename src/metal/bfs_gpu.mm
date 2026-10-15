#include "metal/bfs_gpu.hpp"

#include "util/timer.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace gpu {

namespace {

constexpr uint32_t kInf = 0xFFFFFFFFu;  // INF_DIST in common.h
enum { kStatFrontier = 0, kStatEdges = 1, kStatFound = 2 };

// Keeps the Metal buffers behind a CSR view alive (CSR::storage).
struct BufferStorage {
    id<MTLBuffer> rowPtr, colIdx;
};

void setU32(id<MTLComputeCommandEncoder> enc, uint32_t v, NSUInteger index) {
    [enc setBytes:&v length:sizeof v atIndex:index];
}

void checkCompleted(id<MTLCommandBuffer> cb) {
    if (cb.status != MTLCommandBufferStatusCompleted) {
        std::string msg = cb.error ? cb.error.localizedDescription.UTF8String : "unknown error";
        throw std::runtime_error("GPU BFS command buffer failed: " + msg);
    }
}

} // namespace

GpuBfs::GpuBfs(MetalContext& ctx, const graph::CSR& g) : ctx_(ctx), n_((uint32_t)g.n) {
    if (g.n >= kInf) throw std::runtime_error("GpuBfs: graph too large for uint32 vertex ids");
    // One copy into shared buffers (outside every timed region). The CSR view
    // below then aliases these buffers, so the source arrays can be released.
    rowPtr_ = ctx.buffer(g.row_ptr, (g.n + 1) * sizeof(uint32_t));
    colIdx_ = ctx.buffer(g.col_idx, g.m * sizeof(uint32_t));
    dist_ = ctx.buffer(g.n * sizeof(uint32_t));
    stats_ = ctx.buffer(4 * sizeof(uint32_t));

    view_.n = g.n;
    view_.m = g.m;
    view_.flags = g.flags;
    view_.row_ptr = static_cast<const uint32_t*>(rowPtr_.contents);
    view_.col_idx = static_cast<const uint32_t*>(colIdx_.contents);
    auto holder = std::make_shared<BufferStorage>();
    holder->rowPtr = rowPtr_;
    holder->colIdx = colIdx_;
    view_.storage = std::move(holder);

    init_ = ctx.pipeline("bfs", "bfs_init");
    vertexLevel_ = ctx.pipeline("bfs", "bfs_vertex_level");
    naiveLevel_ = ctx.pipeline("bfs", "bfs_naive_level");
}

void GpuBfs::finish(uint32_t* out) {
    if (out && out != dist()) std::memcpy(out, dist(), (size_t)n_ * sizeof(uint32_t));
}

// Per level: zero the stats (host write into the shared buffer; safe because
// the previous level's command buffer has completed), commit one command
// buffer with the level kernel, wait, read the stats. The wait is the
// per-level host round trip (~111 us measured in Phase 3) that later phases
// try to avoid.
void GpuBfs::runVertex(uint32_t source, uint32_t* out, const GpuBfsOptions& opt,
                       std::vector<util::BfsLevel>* trace) {
    uint32_t* stats = static_cast<uint32_t*>(stats_.contents);
    lastGpuMs_ = 0;
    lastLevels_ = 0;
    for (uint32_t level = 0;; ++level) {
        @autoreleasepool {
            util::Timer t;
            std::memset(stats, 0, 4 * sizeof(uint32_t));
            id<MTLCommandBuffer> cb = [ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (level == 0) {  // initialization rides in the first level's command buffer
                [enc setBuffer:dist_ offset:0 atIndex:0];
                setU32(enc, n_, 1);
                setU32(enc, source, 2);
                MetalContext::dispatch1D(enc, init_, n_, 256);
            }
            [enc setBuffer:rowPtr_ offset:0 atIndex:0];
            [enc setBuffer:colIdx_ offset:0 atIndex:1];
            [enc setBuffer:dist_ offset:0 atIndex:2];
            [enc setBuffer:stats_ offset:0 atIndex:3];
            setU32(enc, n_, 4);
            setU32(enc, level, 5);
            MetalContext::dispatch1D(enc, vertexLevel_, n_, opt.tg);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            checkCompleted(cb);
            lastGpuMs_ += MetalContext::gpuTimeMs(cb);
            ++lastLevels_;
            if (trace)
                trace->push_back({level, stats[kStatFrontier], stats[kStatEdges], 'T', 'G', t.ms()});
            if (stats[kStatFound] == 0) break;  // nothing discovered: next frontier empty
        }
    }
    finish(out);
}

void GpuBfs::runNaive(uint32_t source, uint32_t* out, const GpuBfsOptions& opt,
                      std::vector<util::BfsLevel>* trace) {
    if (!queueA_) {
        queueA_ = ctx_.buffer((size_t)n_ * sizeof(uint32_t));
        queueB_ = ctx_.buffer((size_t)n_ * sizeof(uint32_t));
    }
    uint32_t* stats = static_cast<uint32_t*>(stats_.contents);
    id<MTLBuffer> cur = queueA_, next = queueB_;
    static_cast<uint32_t*>(cur.contents)[0] = source;  // host write; no GPU work in flight
    uint32_t size = 1;
    lastGpuMs_ = 0;
    lastLevels_ = 0;
    for (uint32_t level = 0; size > 0; ++level) {
        @autoreleasepool {
            util::Timer t;
            std::memset(stats, 0, 4 * sizeof(uint32_t));
            id<MTLCommandBuffer> cb = [ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (level == 0) {
                [enc setBuffer:dist_ offset:0 atIndex:0];
                setU32(enc, n_, 1);
                setU32(enc, source, 2);
                MetalContext::dispatch1D(enc, init_, n_, 256);
            }
            [enc setBuffer:rowPtr_ offset:0 atIndex:0];
            [enc setBuffer:colIdx_ offset:0 atIndex:1];
            [enc setBuffer:dist_ offset:0 atIndex:2];
            [enc setBuffer:stats_ offset:0 atIndex:3];
            [enc setBuffer:cur offset:0 atIndex:4];
            [enc setBuffer:next offset:0 atIndex:5];
            setU32(enc, size, 6);
            setU32(enc, level, 7);
            MetalContext::dispatch1D(enc, naiveLevel_, size, opt.tg);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            checkCompleted(cb);
            lastGpuMs_ += MetalContext::gpuTimeMs(cb);
            ++lastLevels_;
            if (trace)
                trace->push_back({level, stats[kStatFrontier], stats[kStatEdges], 'T', 'G', t.ms()});
            size = stats[kStatFound];
            std::swap(cur, next);
        }
    }
    finish(out);
}

} // namespace gpu
