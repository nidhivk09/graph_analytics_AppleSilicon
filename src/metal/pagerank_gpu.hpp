#pragma once
// GPU PageRank variants (PLAN.md 5.2). Objective-C++ only.
//
// Upload once (untimed): out-degrees always, the out-CSR for gpu_push, the
// in-CSR (transpose) for the pull variants. run() then does init plus all
// iterations:
//   fixed mode (tol == 0): every iteration is encoded into ONE command
//     buffer: no host sync at all, so time/iteration is pure GPU work.
//   tol mode: one command buffer + sync per iteration to read the L1 change.
//     Float iterates can settle into a rounding cycle whose L1 change stays
//     above tol (star100k: 1.26e-6 for gpu_pull_adaptive, while double
//     reaches 1e-11), so tol mode also stops when the L1 change has not
//     improved for kStagnation iterations (floorStop).
#include "graph/csr.hpp"
#include "metal/context.hpp"
#include "metal/primitives.hpp"

#include <memory>
#include <string>
#include <vector>

namespace gpu {

enum class PrVariant { Push, PullScalar, PullVector, PullAdaptive };

struct GpuPrOptions {
    float damping = 0.85f;
    int iters = 20;          // fixed mode: iterations; tol mode: cap
    double tol = 0;          // > 0: stop when the L1 change < tol
    uint32_t tg = 256;       // threadgroup size of the main kernels (multiple of 32)
    uint32_t smallMax = 16;  // adaptive: in-degree < smallMax -> small bin
    uint32_t largeMin = 256; // adaptive: in-degree >= largeMin -> threadgroup per vertex
};

struct GpuPrResult {
    int iters = 0;
    double lastDelta = -1;   // tol mode only
    bool floorStop = false;  // tol mode: stopped because the L1 change stagnated above tol
    double gpuMs = 0;        // kernel time, summed over command buffers
};

class GpuPageRank {
public:
    // `out` may be empty when !needOut, `in` when !needIn.
    GpuPageRank(MetalContext& ctx, const graph::CSR& out, const graph::CSR& in,
                const std::vector<uint32_t>& outdeg, bool needOut, bool needIn);

    // Bin vertices by in-degree for gpu_pull_adaptive (preprocessing, done
    // once per threshold pair; run() calls it if needed).
    void prepareAdaptive(uint32_t smallMax, uint32_t largeMin);

    GpuPrResult run(PrVariant v, const GpuPrOptions& opt);
    const float* ranks() const { return static_cast<const float*>(r_[cur_].contents); }

private:
    struct Params { uint32_t n; float damping, base, invN; uint32_t size; };
    void encodeIteration(id<MTLComputeCommandEncoder> enc, PrVariant v, const GpuPrOptions& opt,
                         bool withDelta);

    MetalContext& ctx_;
    uint32_t n_;
    Params params_{};
    id<MTLBuffer> rowOut_, colOut_, rowIn_, colIn_, outdeg_;
    id<MTLBuffer> r_[2], contrib_, acc_, accLo_, partial_, scalars_;
    id<MTLBuffer> bins_[3];
    uint32_t binCount_[3] = {0, 0, 0};
    uint32_t binSmall_ = 0, binLarge_ = 0;  // thresholds the bins were built for
    bool binsReady_ = false;
    int cur_ = 0;
    size_t reduceGroups_;
    id<MTLComputePipelineState> init_, contribK_, diff_, reduce_, push_, pushFin_, scalar_, vector_,
        small_, medium_, large_;
    std::unique_ptr<Primitives> prims_;
};

PrVariant parsePrVariant(const std::string& name);  // gpu_push, gpu_pull_scalar, ...

} // namespace gpu
