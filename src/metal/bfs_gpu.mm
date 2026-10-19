#include "metal/bfs_gpu.hpp"

#include "cpu/bfs_steps.hpp"
#include "util/timer.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace gpu {

namespace steps = cpu::steps;

namespace {

constexpr uint32_t kInf = 0xFFFFFFFFu;  // INF_DIST in common.h
enum { kStatFrontier = 0, kStatEdges = 1, kStatFound = 2, kStatNextEdges = 3, kStatBin0 = 4 };
constexpr size_t kStatsWords = 8;

// Mirrors LevelParams in bfs.metal.
struct LevelParams {
    uint32_t size, level, smallMax, largeMin, total, ept;
};

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
    stats_ = ctx.buffer(kStatsWords * sizeof(uint32_t));

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
    binSmall_ = ctx.pipeline("bfs", "bfs_binned_small");
    binMedium_ = ctx.pipeline("bfs", "bfs_binned_medium");
    binLarge_ = ctx.pipeline("bfs", "bfs_binned_large");
    edgeExpand_ = ctx.pipeline("bfs", "bfs_edge_expand");
    queueDegrees_ = ctx.pipeline("bfs", "bfs_queue_degrees");
    bottomUp_ = ctx.pipeline("bfs", "bfs_bottom_up");
    prims_ = std::make_unique<Primitives>(ctx);
}

uint32_t GpuBfs::checkTg(uint32_t tg) {
    // Simdgroup-per-vertex and the aggregated appends assume whole simdgroups.
    if (tg < 32 || tg > 1024 || tg % 32 != 0)
        throw std::runtime_error("threadgroup size must be a multiple of 32 in [32, 1024]");
    return tg;
}

id<MTLCommandBuffer> GpuBfs::beginLevel(uint32_t level, uint32_t source,
                                        id<MTLComputeCommandEncoder>* enc) {
    std::memset(stats_.contents, 0, kStatsWords * sizeof(uint32_t));
    id<MTLCommandBuffer> cb = [ctx_.queue() commandBuffer];
    *enc = [cb computeCommandEncoder];
    if (level == 0) {
        [*enc setBuffer:dist_ offset:0 atIndex:0];
        setU32(*enc, n_, 1);
        setU32(*enc, source, 2);
        MetalContext::dispatch1D(*enc, init_, n_, 256);
    }
    return cb;
}

void GpuBfs::endLevel(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    checkCompleted(cb);
    lastGpuMs_ += MetalContext::gpuTimeMs(cb);
    ++lastLevels_;
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
            id<MTLComputeCommandEncoder> enc;  // level 0's buffer also initializes dist
            id<MTLCommandBuffer> cb = beginLevel(level, source, &enc);
            [enc setBuffer:rowPtr_ offset:0 atIndex:0];
            [enc setBuffer:colIdx_ offset:0 atIndex:1];
            [enc setBuffer:dist_ offset:0 atIndex:2];
            [enc setBuffer:stats_ offset:0 atIndex:3];
            setU32(enc, n_, 4);
            setU32(enc, level, 5);
            MetalContext::dispatch1D(enc, vertexLevel_, n_, checkTg(opt.tg));
            endLevel(cb, enc);
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
            id<MTLComputeCommandEncoder> enc;
            id<MTLCommandBuffer> cb = beginLevel(level, source, &enc);
            [enc setBuffer:rowPtr_ offset:0 atIndex:0];
            [enc setBuffer:colIdx_ offset:0 atIndex:1];
            [enc setBuffer:dist_ offset:0 atIndex:2];
            [enc setBuffer:stats_ offset:0 atIndex:3];
            [enc setBuffer:cur offset:0 atIndex:4];
            [enc setBuffer:next offset:0 atIndex:5];
            setU32(enc, size, 6);
            setU32(enc, level, 7);
            MetalContext::dispatch1D(enc, naiveLevel_, size, checkTg(opt.tg));
            endLevel(cb, enc);
            if (trace)
                trace->push_back({level, stats[kStatFrontier], stats[kStatEdges], 'T', 'G', t.ms()});
            size = stats[kStatFound];
            std::swap(cur, next);
        }
    }
    finish(out);
}

void GpuBfs::runBinned(uint32_t source, uint32_t* out, const GpuBfsOptions& opt,
                       std::vector<util::BfsLevel>* trace) {
    const uint32_t tg = checkTg(opt.tg);
    if (!bins_[0][0])
        for (auto& set : bins_)
            for (auto& q : set) q = ctx_.buffer((size_t)n_ * sizeof(uint32_t));
    const uint32_t* stats = static_cast<const uint32_t*>(stats_.contents);
    auto bin = [&](uint32_t deg) { return deg < opt.smallMax ? 0 : (deg < opt.largeMin ? 1 : 2); };

    // Level 0 frontier = {source}, placed in its bin by the host.
    uint32_t count[3] = {0, 0, 0};
    const uint32_t srcDeg = view_.degree(source);
    static_cast<uint32_t*>(bins_[0][bin(srcDeg)].contents)[0] = source;
    count[bin(srcDeg)] = 1;
    uint64_t edges = srcDeg;  // edges examined this level (known from the previous level)
    int cur = 0;
    lastGpuMs_ = 0;
    lastLevels_ = 0;

    for (uint32_t level = 0; count[0] + count[1] + count[2] > 0; ++level) {
        @autoreleasepool {
            util::Timer t;
            id<MTLComputeCommandEncoder> enc;
            id<MTLCommandBuffer> cb = beginLevel(level, source, &enc);
            [enc setBuffer:rowPtr_ offset:0 atIndex:0];
            [enc setBuffer:colIdx_ offset:0 atIndex:1];
            [enc setBuffer:dist_ offset:0 atIndex:2];
            [enc setBuffer:stats_ offset:0 atIndex:3];
            [enc setBuffer:bins_[cur ^ 1][0] offset:0 atIndex:5];
            [enc setBuffer:bins_[cur ^ 1][1] offset:0 atIndex:6];
            [enc setBuffer:bins_[cur ^ 1][2] offset:0 atIndex:7];
            LevelParams p{0, level, opt.smallMax, opt.largeMin, 0, 0};
            // Three dispatches, one per bin, in one serial encoder. Empty bins are skipped.
            for (int b = 0; b < 3; ++b) {
                if (count[b] == 0) continue;
                p.size = count[b];
                [enc setBuffer:bins_[cur][b] offset:0 atIndex:4];
                [enc setBytes:&p length:sizeof p atIndex:8];
                if (b == 0) {
                    MetalContext::dispatch1D(enc, binSmall_, count[b], tg);
                } else if (b == 1) {
                    MetalContext::dispatch1D(enc, binMedium_, (size_t)count[b] * 32, tg);
                } else {
                    // Clamp to the pipeline's limit (register pressure can lower it).
                    NSUInteger tgL = std::min<NSUInteger>(tg, binLarge_.maxTotalThreadsPerThreadgroup);
                    [enc setComputePipelineState:binLarge_];
                    [enc dispatchThreadgroups:MTLSizeMake(count[b], 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tgL, 1, 1)];
                }
            }
            endLevel(cb, enc);
            if (trace)
                trace->push_back({level, (uint64_t)count[0] + count[1] + count[2], edges, 'T', 'G', t.ms()});
            for (int b = 0; b < 3; ++b) count[b] = stats[kStatBin0 + b];
            edges = stats[kStatNextEdges];
            cur ^= 1;
        }
    }
    finish(out);
}

uint32_t GpuBfs::edgeTg(const GpuBfsOptions& opt) const {
    if (opt.ept == 0) throw std::runtime_error("ept must be >= 1");
    // The expand kernel's chunk size is tg * ept on both sides: use the
    // pipeline-clamped threadgroup size for the grid too.
    return (uint32_t)std::min<NSUInteger>(checkTg(opt.tg), edgeExpand_.maxTotalThreadsPerThreadgroup);
}

void GpuBfs::ensureEdgeBuffers() {
    if (!queueA_) {
        queueA_ = ctx_.buffer((size_t)n_ * sizeof(uint32_t));
        queueB_ = ctx_.buffer((size_t)n_ * sizeof(uint32_t));
    }
    if (!degA_) {
        degA_ = ctx_.buffer((size_t)n_ * sizeof(uint32_t));
        degB_ = ctx_.buffer((size_t)n_ * sizeof(uint32_t));
    }
}

void GpuBfs::encodeEdgeLevel(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> q, id<MTLBuffer> deg,
                             id<MTLBuffer> qNext, id<MTLBuffer> degNext, uint32_t size,
                             uint64_t total, uint32_t level, uint32_t tg, uint32_t ept) {
    // Degrees -> exclusive offsets, in place (they are not needed afterwards).
    prims_->scanExclusive(enc, deg, deg, size);
    [enc setBuffer:rowPtr_ offset:0 atIndex:0];
    [enc setBuffer:colIdx_ offset:0 atIndex:1];
    [enc setBuffer:dist_ offset:0 atIndex:2];
    [enc setBuffer:stats_ offset:0 atIndex:3];
    [enc setBuffer:q offset:0 atIndex:4];
    [enc setBuffer:deg offset:0 atIndex:5];
    [enc setBuffer:qNext offset:0 atIndex:6];
    [enc setBuffer:degNext offset:0 atIndex:7];
    LevelParams p{size, level, 0, 0, (uint32_t)total, ept};
    [enc setBytes:&p length:sizeof p atIndex:8];
    const uint64_t chunk = (uint64_t)tg * ept;
    [enc setComputePipelineState:edgeExpand_];
    [enc dispatchThreadgroups:MTLSizeMake((total + chunk - 1) / chunk, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

void GpuBfs::runEdge(uint32_t source, uint32_t* out, const GpuBfsOptions& opt,
                     std::vector<util::BfsLevel>* trace) {
    const uint32_t tg = edgeTg(opt);
    ensureEdgeBuffers();
    const uint32_t* stats = static_cast<const uint32_t*>(stats_.contents);
    id<MTLBuffer> q = queueA_, qNext = queueB_, deg = degA_, degNext = degB_;
    static_cast<uint32_t*>(q.contents)[0] = source;
    static_cast<uint32_t*>(deg.contents)[0] = view_.degree(source);
    uint32_t size = 1;
    uint64_t total = view_.degree(source);
    lastGpuMs_ = 0;
    lastLevels_ = 0;

    for (uint32_t level = 0; size > 0; ++level) {
        @autoreleasepool {
            util::Timer t;
            id<MTLComputeCommandEncoder> enc;
            id<MTLCommandBuffer> cb = beginLevel(level, source, &enc);
            if (total > 0) encodeEdgeLevel(enc, q, deg, qNext, degNext, size, total, level, tg, opt.ept);
            endLevel(cb, enc);
            if (trace) trace->push_back({level, size, total, 'T', 'G', t.ms()});
            size = stats[kStatFound];
            total = stats[kStatNextEdges];
            std::swap(q, qNext);
            std::swap(deg, degNext);
        }
    }
    finish(out);
}

// Direction-optimizing BFS, optionally hybrid (Phase 7).
//
// Switching rule copied from cpu::bfsDO (GAP): go bottom-up when the
// frontier's edges (scout) exceed the edges not yet explored top-down
// (toCheck) / alpha; stay bottom-up while the frontier grows or is larger
// than n / beta; after a bottom-up phase GAP sets scout = 1 (kept, so CPU and
// GPU make identical decisions).
//
// Device per level (hybrid): a level runs on the CPU (OpenMP, directly on the
// shared buffers) if its work is below `threshold`. Work = the frontier's
// edges for a top-down level (known exactly from the previous level), and
// the unexplored edges toCheck for a bottom-up level (an upper bound: bottom-
// up work is only known afterwards). threshold 0 = all GPU (gpu_do),
// UINT64_MAX = all CPU. Each GPU level is one command buffer + one sync; a
// CPU level needs no sync at all. A format conversion (queue <-> bitmap) is
// done by whichever device runs the first level in the new direction.
// Correctness across devices relies on unified memory: after
// waitUntilCompleted the CPU sees the GPU's writes, and CPU writes made
// before a commit are visible to that command buffer.
void GpuBfs::runDirOpt(uint32_t source, uint32_t* out, const GpuBfsOptions& opt,
                       std::vector<util::BfsLevel>* trace, uint64_t threshold) {
    const uint32_t tg = edgeTg(opt);
    const uint32_t tgBu = checkTg(opt.tg);
    ensureEdgeBuffers();
    const size_t words = ((size_t)n_ + 31) / 32;
    if (!frontBm_) {
        frontBm_ = ctx_.buffer(words * sizeof(uint32_t));
        nextBm_ = ctx_.buffer(words * sizeof(uint32_t));
        bmCount_ = ctx_.buffer(sizeof(uint32_t));
    }
    const uint32_t* stats = static_cast<const uint32_t*>(stats_.contents);
    auto u32 = [](id<MTLBuffer> b) { return static_cast<uint32_t*>(b.contents); };
    id<MTLBuffer> q = queueA_, qNext = queueB_, deg = degA_, degNext = degB_;
    id<MTLBuffer> front = frontBm_, next = nextBm_;
    u32(q)[0] = source;
    u32(deg)[0] = view_.degree(source);

    uint64_t size = 1;                       // frontier vertices (queue or bitmap)
    uint64_t total = view_.degree(source);   // frontier edges (exact; sizes the TD dispatch)
    uint64_t scout = total;                  // heuristic's m_f (GAP resets it to 1 after BU)
    uint64_t toCheck = view_.m;              // heuristic's m_u
    bool bottomUp = false, toBitmap = false, toQueue = false;
    lastGpuMs_ = 0;
    lastLevels_ = 0;

    for (uint32_t level = 0; size > 0; ++level) {
        @autoreleasepool {
            if (!bottomUp && (double)scout > (double)toCheck / opt.alpha) {
                bottomUp = true;
                toBitmap = true;
            }
            const uint64_t work = bottomUp ? toCheck : total;
            const bool onCpu = work < threshold;
            util::Timer t;
            steps::StepOut r;  // this level's counters, from either device

            if (!bottomUp) toCheck -= std::min(toCheck, scout);
            if (onCpu) {
                if (level == 0) steps::initDist(dist(), n_, source, opt.cpuThreads);
                if (bottomUp) {
                    if (toBitmap) steps::queueToBitmap(u32(q), size, u32(front), n_, opt.cpuThreads);
                    r = steps::bottomUp(view_, dist(), u32(front), u32(next), level, opt.cpuThreads);
                } else {
                    if (toQueue) steps::bitmapToQueue(view_, u32(front), u32(q), u32(deg), opt.cpuThreads);
                    r = steps::topDown(view_, dist(), u32(q), size, total, u32(qNext), u32(degNext),
                                       level, opt.cpuThreads);
                }
            } else {
                id<MTLComputeCommandEncoder> enc;
                id<MTLCommandBuffer> cb = beginLevel(level, source, &enc);  // level 0: + dist init
                if (bottomUp) {
                    if (toBitmap) prims_->queueToBitmap(enc, q, size, front, n_);
                    [enc setBuffer:rowPtr_ offset:0 atIndex:0];
                    [enc setBuffer:colIdx_ offset:0 atIndex:1];
                    [enc setBuffer:dist_ offset:0 atIndex:2];
                    [enc setBuffer:front offset:0 atIndex:3];
                    [enc setBuffer:next offset:0 atIndex:4];
                    [enc setBuffer:stats_ offset:0 atIndex:5];
                    setU32(enc, n_, 6);
                    setU32(enc, level, 7);
                    MetalContext::dispatch1D(enc, bottomUp_, n_, tgBu);
                } else {
                    if (toQueue) {
                        // Bitmap -> queue (unordered), then the degrees gpu_edge scans.
                        prims_->bitmapToQueue(enc, front, n_, q, bmCount_);
                        [enc setBuffer:rowPtr_ offset:0 atIndex:0];
                        [enc setBuffer:q offset:0 atIndex:1];
                        [enc setBuffer:deg offset:0 atIndex:2];
                        setU32(enc, (uint32_t)size, 3);
                        MetalContext::dispatch1D(enc, queueDegrees_, size, 256);
                    }
                    if (total > 0)
                        encodeEdgeLevel(enc, q, deg, qNext, degNext, (uint32_t)size, total, level, tg, opt.ept);
                }
                endLevel(cb, enc);
                r.found = stats[kStatFound];
                r.examined = bottomUp ? stats[kStatEdges] : total;
                r.nextEdges = stats[kStatNextEdges];
            }
            toBitmap = toQueue = false;
            if (trace)
                trace->push_back({level, size, bottomUp ? r.examined : total, bottomUp ? 'B' : 'T',
                                  onCpu ? 'C' : 'G', t.ms()});

            const uint64_t oldSize = size;
            size = r.found;
            total = r.nextEdges;
            if (bottomUp) {
                std::swap(front, next);
                if (!(size >= oldSize || (double)size > (double)n_ / opt.beta)) {
                    bottomUp = false;
                    toQueue = true;
                    scout = 1;
                }
            } else {
                scout = total;
                std::swap(q, qNext);
                std::swap(deg, degNext);
            }
        }
    }
    finish(out);
}

} // namespace gpu
