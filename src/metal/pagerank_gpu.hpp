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

// PullGS (Phase 12): in-place (Gauss-Seidel-style) pull; see prepareGS.
enum class PrVariant { Push, PullScalar, PullVector, PullAdaptive, PullGS };

struct GpuPrOptions {
    float damping = 0.85f;
    int iters = 20;          // fixed mode: iterations; tol mode: cap
    double tol = 0;          // > 0: stop when the L1 change < tol
    uint32_t tg = 256;       // threadgroup size of the main kernels (multiple of 32)
    uint32_t smallMax = 16;  // adaptive: in-degree < smallMax -> small bin
    uint32_t largeMin = 256; // adaptive: in-degree >= largeMin -> threadgroup per vertex
    bool inPlace = false;    // any pull variant: update contrib in place (Phase 12 experiments)
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

    // gpu_pull_gs update order (preprocessing, once). A GPU sweep updates
    // tens of thousands of vertices at a time, so an in-place update only
    // helps when a vertex's in-neighbors were updated in an EARLIER part of
    // the sweep. Two orders, chosen per graph:
    //  - colored: up to kGsColors rounds of Jones-Plassmann (a vertex joins
    //    the round's set if its hash priority beats every uncolored
    //    in-neighbor), one dispatch per set, leftovers last. Every edge then
    //    crosses sets, so one endpoint always sees a fresh value. Used when
    //    the sets cover >= 95% of vertices (low-degree graphs: roadNet-CA).
    //    The sets are interleaved in vertex order, so dispatching them over
    //    the original layout made every dispatch touch every cache line of
    //    the CSR (roadNet: 4.2 vs 1.3 ms/iteration). Colored mode therefore
    //    runs on a private relabeled copy of the in-CSR in which each set is
    //    a contiguous id range (ids keep their order inside a set);
    //    ranks() maps the result back to the caller's ids.
    //  - natural: one in-place gpu_pull_vector dispatch in vertex order;
    //    GPU waves run roughly in id order, so later waves see earlier ones.
    // Natural order uses gpu_pull_vector because it is the fastest Jacobi
    // kernel on the graphs that reach this branch (social/web); in-place
    // degree-bin order (gpu_pull_adaptive --inplace) needs about the same
    // number of sweeps (scripts/bench_gs.sh ablation).
    // Limitation: vertices updated in the same wave still read each other's
    // old values, so local structures such as a 2-cycle with adjacent ids
    // converge only at the Jacobi rate while the rest of the graph is faster;
    // the global L1-change stopping test then stops too early for them.
    void prepareGS();
    bool gsColored() const { return gsColored_; }
    size_t gsSets() const { return gsBlocks_.size(); }

    GpuPrResult run(PrVariant v, const GpuPrOptions& opt);
    const float* ranks() const;  // in the caller's vertex ids

private:
    struct Params { uint32_t n; float damping, base, invN; uint32_t size, inplace; };
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
    bool gsReady_ = false, gsColored_ = false;
    std::vector<id<MTLBuffer>> gsBlocks_;
    std::vector<uint32_t> gsBlockSize_;
    // Colored mode: relabeled in-CSR / out-degrees, newId[old], and the
    // ranks mapped back to old ids (filled lazily by ranks(), untimed).
    id<MTLBuffer> rowGS_, colGS_, outdegGS_;
    std::vector<uint32_t> gsNewId_;
    bool gsLastRun_ = false;
    mutable std::vector<float> gsRanks_;
    mutable bool gsRanksValid_ = false;
    int cur_ = 0;
    size_t reduceGroups_;
    id<MTLComputePipelineState> init_, contribK_, diff_, reduce_, push_, pushFin_, scalar_, vector_,
        small_, medium_, large_, sum_, scale_;
    std::unique_ptr<Primitives> prims_;
};

PrVariant parsePrVariant(const std::string& name);  // gpu_push, gpu_pull_scalar, ..., gpu_pull_gs

} // namespace gpu
