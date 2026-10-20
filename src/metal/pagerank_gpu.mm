#include "metal/pagerank_gpu.hpp"

#include <algorithm>
#include <stdexcept>

namespace gpu {

namespace {
constexpr size_t kTG = 256;
constexpr size_t kMaxGroups = 1024;  // reduction pass 2 is one 1024-thread group
enum { kDangling = 0, kDelta = 1, kSum = 2, kMaxRel = 3 };

void checkCompleted(id<MTLCommandBuffer> cb) {
    if (cb.status != MTLCommandBufferStatusCompleted) {
        std::string msg = cb.error ? cb.error.localizedDescription.UTF8String : "unknown error";
        throw std::runtime_error("GPU PageRank command buffer failed: " + msg);
    }
}
} // namespace

PrVariant parsePrVariant(const std::string& s) {
    if (s == "gpu_push") return PrVariant::Push;
    if (s == "gpu_pull_scalar") return PrVariant::PullScalar;
    if (s == "gpu_pull_vector") return PrVariant::PullVector;
    if (s == "gpu_pull_adaptive") return PrVariant::PullAdaptive;
    if (s == "gpu_pull_gs") return PrVariant::PullGS;
    throw std::runtime_error("unknown GPU PageRank variant '" + s + "'");
}

GpuPageRank::GpuPageRank(MetalContext& ctx, const graph::CSR& out, const graph::CSR& in,
                         const std::vector<uint32_t>& outdeg, bool needOut, bool needIn)
    : ctx_(ctx), n_((uint32_t)outdeg.size()) {
    if (needOut) {
        rowOut_ = ctx.buffer(out.row_ptr, (out.n + 1) * 4);
        colOut_ = ctx.buffer(out.col_idx, out.m * 4);
    }
    if (needIn) {
        rowIn_ = ctx.buffer(in.row_ptr, (in.n + 1) * 4);
        colIn_ = ctx.buffer(in.col_idx, in.m * 4);
    }
    outdeg_ = ctx.buffer(outdeg.data(), outdeg.size() * 4);
    for (auto& b : r_) b = ctx.buffer((size_t)n_ * 4);
    contrib_ = ctx.buffer((size_t)n_ * 4);
    if (needOut) {
        acc_ = ctx.buffer((size_t)n_ * 4);
        accLo_ = ctx.buffer((size_t)n_ * 4);  // compensation terms (see pr_push)
    }
    reduceGroups_ = std::min(kMaxGroups, std::max<size_t>(1, ((size_t)n_ + kTG - 1) / kTG));
    partial_ = ctx.buffer(reduceGroups_ * 4);
    partialMax_ = ctx.buffer(reduceGroups_ * 4);
    scalars_ = ctx.buffer(4 * 4);

    init_ = ctx.pipeline("pagerank", "pr_init");
    contribK_ = ctx.pipeline("pagerank", "pr_contrib");
    diff_ = ctx.pipeline("pagerank", "pr_diff");
    reduce_ = ctx.pipeline("pagerank", "pr_reduce");
    push_ = ctx.pipeline("pagerank", "pr_push");
    pushFin_ = ctx.pipeline("pagerank", "pr_push_finalize");
    scalar_ = ctx.pipeline("pagerank", "pr_pull_scalar");
    vector_ = ctx.pipeline("pagerank", "pr_pull_vector");
    small_ = ctx.pipeline("pagerank", "pr_pull_small");
    medium_ = ctx.pipeline("pagerank", "pr_pull_medium");
    large_ = ctx.pipeline("pagerank", "pr_pull_large");
    sum_ = ctx.pipeline("pagerank", "pr_sum");
    scale_ = ctx.pipeline("pagerank", "pr_scale");
    reduceMax_ = ctx.pipeline("pagerank", "pr_reduce_max");
    prims_ = std::make_unique<Primitives>(ctx);
}

void GpuPageRank::prepareAdaptive(uint32_t smallMax, uint32_t largeMin) {
    if (binsReady_ && binSmall_ == smallMax && binLarge_ == largeMin) return;
    if (!rowIn_) throw std::runtime_error("gpu_pull_adaptive needs the in-CSR");
    const uint32_t* rp = static_cast<const uint32_t*>(rowIn_.contents);
    std::vector<uint32_t> lists[3];
    for (uint32_t v = 0; v < n_; ++v) {
        uint32_t d = rp[v + 1] - rp[v];
        lists[d < smallMax ? 0 : (d < largeMin ? 1 : 2)].push_back(v);
    }
    for (int b = 0; b < 3; ++b) {
        binCount_[b] = (uint32_t)lists[b].size();
        bins_[b] = ctx_.buffer(lists[b].data(), lists[b].size() * 4);
    }
    binSmall_ = smallMax;
    binLarge_ = largeMin;
    binsReady_ = true;
}

void GpuPageRank::prepareGS(uint32_t pairMaxOutdeg) {
    if (gsReady_ && gsPairK_ == pairMaxOutdeg) return;
    gsPairK_ = pairMaxOutdeg;
    if (!rowIn_) throw std::runtime_error("gpu_pull_gs needs the in-CSR");
    constexpr int kGsColors = 8;
    constexpr double kMinCoverage = 0.95;
    const uint32_t* rp = static_cast<const uint32_t*>(rowIn_.contents);
    const uint32_t* ci = static_cast<const uint32_t*>(colIn_.contents);
    // Deterministic pseudo-random priorities (ties broken by id).
    auto prio = [](uint32_t v) {
        uint64_t x = v * 0x9E3779B97F4A7C15ull;
        x ^= x >> 31;
        return (x << 32) | v;
    };
    std::vector<uint8_t> color(n_, 0xFF);
    std::vector<std::vector<uint32_t>> sets;
    size_t covered = 0;
    for (int c = 0; c < kGsColors && covered < n_; ++c) {
        std::vector<uint32_t> pick;
        for (uint32_t v = 0; v < n_; ++v) {
            if (color[v] != 0xFF) continue;
            bool best = true;
            for (uint32_t e = rp[v]; e < rp[v + 1] && best; ++e) {
                uint32_t u = ci[e];
                best = u == v || color[u] != 0xFF || prio(u) < prio(v);
            }
            if (best) pick.push_back(v);
        }
        for (uint32_t v : pick) color[v] = (uint8_t)c;  // after the round: sets are decided on uncolored state
        covered += pick.size();
        sets.push_back(std::move(pick));
    }
    gsColored_ = covered >= kMinCoverage * n_;
    gsBlocks_.clear();
    gsBlockSize_.clear();
    if (gsColored_) {
        std::vector<uint32_t> rest;
        for (uint32_t v = 0; v < n_; ++v)
            if (color[v] == 0xFF) rest.push_back(v);
        if (!rest.empty()) sets.push_back(std::move(rest));
        // Color-major relabeling: set 0 gets ids [0, |set 0|), and so on.
        std::vector<uint32_t> order;
        order.reserve(n_);
        gsNewId_.assign(n_, 0);
        for (auto& s : sets) {
            if (s.empty()) continue;
            std::vector<uint32_t> ids(s.size());
            for (size_t i = 0; i < s.size(); ++i) {
                gsNewId_[s[i]] = (uint32_t)(order.size() + i);
                ids[i] = (uint32_t)(order.size() + i);
            }
            order.insert(order.end(), s.begin(), s.end());
            gsBlockSize_.push_back((uint32_t)s.size());
            gsBlocks_.push_back(ctx_.buffer(ids.data(), ids.size() * 4));  // the set's (new) ids
        }
        const uint32_t* od = static_cast<const uint32_t*>(outdeg_.contents);
        std::vector<uint32_t> row(n_ + 1, 0), col(rp[n_]), deg(n_);
        for (uint32_t i = 0; i < n_; ++i) {
            uint32_t v = order[i];
            row[i + 1] = row[i] + (rp[v + 1] - rp[v]);
            deg[i] = od[v];
            uint32_t* out = col.data() + row[i];
            for (uint32_t e = rp[v]; e < rp[v + 1]; ++e) *out++ = gsNewId_[ci[e]];
            std::sort(col.data() + row[i], out);  // ascending gathers, as in the original lists
        }
        rowGS_ = ctx_.buffer(row.data(), row.size() * 4);
        colGS_ = col.empty() ? ctx_.buffer(4) : ctx_.buffer(col.data(), col.size() * 4);  // m = 0: placeholder
        outdegGS_ = ctx_.buffer(deg.data(), deg.size() * 4);
    }
    // Natural order: defer one vertex of each strongly coupled mutual pair.
    // u -> v exists iff u is in v's in-list (sorted: binary search). Greedy in
    // id order: v is deferred if it has a lower-id partner and none of its
    // lower-id partners is deferred, so the deferred set contains no pair
    // (otherwise partners would again run concurrently in the second
    // dispatch: web-Google 873801 with both 750372 and 875012).
    gsDeferredCount_ = 0;
    gsNatASize_ = gsNatBSize_ = 0;
    if (!gsColored_ && pairMaxOutdeg > 0) {
        const uint32_t* od = static_cast<const uint32_t*>(outdeg_.contents);
        std::vector<uint8_t> deferred(n_, 0);
        for (uint32_t v = 0; v < n_; ++v) {
            if (od[v] == 0 || od[v] > pairMaxOutdeg) continue;
            bool hasPartner = false, partnerDeferred = false;
            for (uint32_t e = rp[v]; e < rp[v + 1] && !partnerDeferred; ++e) {
                uint32_t u = ci[e];  // u -> v
                if (u >= v || od[u] == 0 || od[u] > pairMaxOutdeg) continue;
                if (std::binary_search(ci + rp[u], ci + rp[u + 1], v)) {  // v -> u too
                    hasPartner = true;
                    partnerDeferred = deferred[u];
                }
            }
            deferred[v] = hasPartner && !partnerDeferred;
        }
        std::vector<uint32_t> a, b;
        for (uint32_t v = 0; v < n_; ++v) (deferred[v] ? b : a).push_back(v);
        if (!b.empty()) {
            gsDeferredCount_ = b.size();
            gsNatASize_ = (uint32_t)a.size();
            gsNatBSize_ = (uint32_t)b.size();
            gsNatA_ = ctx_.buffer(a.data(), a.size() * 4);
            gsNatB_ = ctx_.buffer(b.data(), b.size() * 4);
        }
    }
    gsReady_ = true;
}

const float* GpuPageRank::ranks() const {
    const float* r = static_cast<const float*>(r_[cur_].contents);
    if (!gsLastRun_) return r;
    if (!gsRanksValid_) {
        gsRanks_.resize(n_);
        for (uint32_t v = 0; v < n_; ++v) gsRanks_[v] = r[gsNewId_[v]];
        gsRanksValid_ = true;
    }
    return gsRanks_.data();
}

void GpuPageRank::encodeIteration(id<MTLComputeCommandEncoder> enc, PrVariant v,
                                  const GpuPrOptions& opt, bool withDelta) {
    id<MTLBuffer> r = r_[cur_], rNew = r_[cur_ ^ 1];
    // gpu_pull_gs colored mode works in its relabeled id space throughout.
    const bool relabeled = v == PrVariant::PullGS && gsColored_;
    id<MTLBuffer> outdeg = relabeled ? outdegGS_ : outdeg_;
    const uint32_t slotD = kDangling, slotX = kDelta, slotS = kSum, groups = (uint32_t)reduceGroups_;
    auto reduceTo = [&](const uint32_t* slot) {
        [enc setComputePipelineState:reduce_];
        [enc setBuffer:partial_ offset:0 atIndex:0];
        [enc setBuffer:scalars_ offset:0 atIndex:1];
        [enc setBytes:&groups length:4 atIndex:2];
        [enc setBytes:slot length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    };

    // (1) contrib + dangling sum
    [enc setComputePipelineState:contribK_];
    [enc setBuffer:r offset:0 atIndex:0];
    [enc setBuffer:outdeg offset:0 atIndex:1];
    [enc setBuffer:contrib_ offset:0 atIndex:2];
    [enc setBuffer:partial_ offset:0 atIndex:3];
    [enc setBytes:&params_ length:sizeof params_ atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
    reduceTo(&slotD);

    // (2) main kernel -> rNew
    const uint32_t tg = opt.tg;
    if (v == PrVariant::Push) {
        [enc setBuffer:rowOut_ offset:0 atIndex:0];
        [enc setBuffer:colOut_ offset:0 atIndex:1];
        [enc setBuffer:contrib_ offset:0 atIndex:2];
        [enc setBuffer:acc_ offset:0 atIndex:3];
        [enc setBytes:&params_ length:sizeof params_ atIndex:4];
        [enc setBuffer:accLo_ offset:0 atIndex:5];
        MetalContext::dispatch1D(enc, push_, n_, tg);
        [enc setBuffer:acc_ offset:0 atIndex:0];
        [enc setBuffer:rNew offset:0 atIndex:1];
        [enc setBuffer:scalars_ offset:0 atIndex:2];
        [enc setBytes:&params_ length:sizeof params_ atIndex:3];
        [enc setBuffer:accLo_ offset:0 atIndex:4];
        MetalContext::dispatch1D(enc, pushFin_, n_, 256);
    } else {
        [enc setBuffer:relabeled ? rowGS_ : rowIn_ offset:0 atIndex:0];
        [enc setBuffer:relabeled ? colGS_ : colIn_ offset:0 atIndex:1];
        [enc setBuffer:contrib_ offset:0 atIndex:2];
        [enc setBuffer:rNew offset:0 atIndex:3];
        [enc setBuffer:scalars_ offset:0 atIndex:4];
        [enc setBytes:&params_ length:sizeof params_ atIndex:5];
        [enc setBuffer:outdeg offset:0 atIndex:7];  // pr_store, in-place mode only
        if (relabeled) {
            // One thread-per-vertex dispatch per color set, in order (the
            // encoder's dispatches are serialized, so set k sees set k-1).
            for (size_t b = 0; b < gsBlocks_.size(); ++b) {
                Params p = params_;
                p.size = gsBlockSize_[b];
                [enc setBytes:&p length:sizeof p atIndex:5];
                [enc setBuffer:gsBlocks_[b] offset:0 atIndex:6];
                MetalContext::dispatch1D(enc, small_, gsBlockSize_[b], tg);
            }
        } else if (v == PrVariant::PullGS && gsNatBSize_ > 0) {
            // Natural order with deferred pair partners: simdgroup per vertex
            // (as gpu_pull_vector) over the main list, then the deferred list.
            const std::pair<id<MTLBuffer>, uint32_t> lists[2] = {{gsNatA_, gsNatASize_}, {gsNatB_, gsNatBSize_}};
            for (const auto& [list, size] : lists) {
                Params p = params_;
                p.size = size;
                [enc setBytes:&p length:sizeof p atIndex:5];
                [enc setBuffer:list offset:0 atIndex:6];
                MetalContext::dispatch1D(enc, medium_, (size_t)size * 32, tg);
            }
        } else if (v == PrVariant::PullScalar) {
            MetalContext::dispatch1D(enc, scalar_, n_, tg);
        } else if (v == PrVariant::PullVector || v == PrVariant::PullGS) {
            MetalContext::dispatch1D(enc, vector_, (size_t)n_ * 32, tg);
        } else {
            for (int b = 0; b < 3; ++b) {
                if (binCount_[b] == 0) continue;
                Params p = params_;
                p.size = binCount_[b];
                [enc setBytes:&p length:sizeof p atIndex:5];
                [enc setBuffer:bins_[b] offset:0 atIndex:6];
                if (b == 0) {
                    MetalContext::dispatch1D(enc, small_, binCount_[b], tg);
                } else if (b == 1) {
                    MetalContext::dispatch1D(enc, medium_, (size_t)binCount_[b] * 32, tg);
                } else {
                    NSUInteger t = std::min<NSUInteger>(tg, large_.maxTotalThreadsPerThreadgroup);
                    [enc setComputePipelineState:large_];
                    [enc dispatchThreadgroups:MTLSizeMake(binCount_[b], 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(t, 1, 1)];
                }
            }
        }
    }

    // (2b) in-place updates: renormalize rNew to sum 1 (see pr_sum)
    if (params_.inplace) {
        [enc setComputePipelineState:sum_];
        [enc setBuffer:rNew offset:0 atIndex:0];
        [enc setBuffer:partial_ offset:0 atIndex:1];
        [enc setBytes:&params_ length:sizeof params_ atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        reduceTo(&slotS);
        [enc setBuffer:rNew offset:0 atIndex:0];
        [enc setBuffer:scalars_ offset:0 atIndex:1];
        [enc setBytes:&params_ length:sizeof params_ atIndex:2];
        MetalContext::dispatch1D(enc, scale_, n_, 256);
    }

    // (3) L1 change (tol mode only)
    if (withDelta) {
        [enc setComputePipelineState:diff_];
        [enc setBuffer:r offset:0 atIndex:0];
        [enc setBuffer:rNew offset:0 atIndex:1];
        [enc setBuffer:partial_ offset:0 atIndex:2];
        [enc setBytes:&params_ length:sizeof params_ atIndex:3];
        [enc setBuffer:partialMax_ offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kTG, 1, 1)];
        reduceTo(&slotX);
        const uint32_t slotM = kMaxRel;
        [enc setComputePipelineState:reduceMax_];
        [enc setBuffer:partialMax_ offset:0 atIndex:0];
        [enc setBuffer:scalars_ offset:0 atIndex:1];
        [enc setBytes:&groups length:4 atIndex:2];
        [enc setBytes:&slotM length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    cur_ ^= 1;
}

GpuPrResult GpuPageRank::run(PrVariant v, const GpuPrOptions& opt) {
    if (opt.tg < 32 || opt.tg > 1024 || opt.tg % 32)
        throw std::runtime_error("threadgroup size must be a multiple of 32 in [32, 1024]");
    if (v == PrVariant::Push && !rowOut_) throw std::runtime_error("gpu_push needs the out-CSR");
    if (v != PrVariant::Push && !rowIn_) throw std::runtime_error("pull variants need the in-CSR");
    if (v == PrVariant::PullAdaptive) prepareAdaptive(opt.smallMax, opt.largeMin);
    if (v == PrVariant::PullGS && !gsReady_) prepareGS();  // keep the caller's pair setting

    params_ = {n_, opt.damping, (1.0f - opt.damping) / (float)n_, 1.0f / (float)n_, 0,
               (v == PrVariant::PullGS || opt.inPlace) ? 1u : 0u};
    GpuPrResult res;
    cur_ = 0;
    gsLastRun_ = v == PrVariant::PullGS && gsColored_;
    gsRanksValid_ = false;
    if (n_ == 0) return res;
    const float* scalars = static_cast<const float*>(scalars_.contents);

    auto begin = [&](id<MTLCommandBuffer>* cb) {
        *cb = [ctx_.queue() commandBuffer];
        return [*cb computeCommandEncoder];
    };
    auto end = [&](id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        checkCompleted(cb);
        res.gpuMs += MetalContext::gpuTimeMs(cb);
    };
    auto encodeInit = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:r_[0] offset:0 atIndex:0];
        [enc setBytes:&params_ length:sizeof params_ atIndex:1];
        MetalContext::dispatch1D(enc, init_, n_, 256);
        if (v == PrVariant::Push) {  // 0.0f has all-zero bits
            prims_->fill(enc, acc_, 0, n_);
            prims_->fill(enc, accLo_, 0, n_);
        }
    };

    @autoreleasepool {
        if (opt.tol <= 0) {
            id<MTLCommandBuffer> cb;
            id<MTLComputeCommandEncoder> enc = begin(&cb);
            encodeInit(enc);
            for (int it = 0; it < opt.iters; ++it) encodeIteration(enc, v, opt, false);
            end(cb, enc);
            res.iters = opt.iters;
            return res;
        }
        // Stagnation guard: 5 iterations without a new best L1 change means
        // the float iterate is at its precision floor (exact PageRank deltas
        // shrink monotonically by ~d per iteration once past the start).
        constexpr int kStagnation = 5;
        double best = 1e300;
        int sinceBest = 0;
        for (int it = 0; it < opt.iters; ++it) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb;
                id<MTLComputeCommandEncoder> enc = begin(&cb);
                if (it == 0) encodeInit(enc);
                encodeIteration(enc, v, opt, true);
                end(cb, enc);
                res.iters = it + 1;
                res.lastDelta = scalars[kDelta];
                res.lastMaxRel = scalars[kMaxRel];
                if (res.lastDelta < opt.tol && (opt.tolRel <= 0 || res.lastMaxRel < opt.tolRel)) break;
                if (res.lastDelta < best) { best = res.lastDelta; sinceBest = 0; }
                else if (++sinceBest >= kStagnation) { res.floorStop = true; break; }
            }
        }
    }
    return res;
}

} // namespace gpu
