#pragma once
// GPU BFS variants (PLAN.md 5.1). Objective-C++ only.
//
// GpuBfs uploads a symmetric CSR once into shared-storage Metal buffers and
// exposes a CSR *view* of those same buffers (graph()). Callers should drop
// their own copy and use the view: CPU code (the validation reference, and
// later the hybrid variant) then reads the exact bytes the GPU reads, and the
// graph is resident only once (Orkut's col_idx alone is ~0.9 GB).
//
// Each run writes dist into a shared buffer (dist()). Passing that pointer as
// `out` makes the result zero-copy; any other pointer gets a copy.
#include "graph/csr.hpp"
#include "metal/context.hpp"
#include "metal/primitives.hpp"
#include "util/trace.hpp"

#include <memory>

#include <vector>

namespace gpu {

struct GpuBfsOptions {
    uint32_t tg = 256;          // threadgroup size for the level kernels (multiple of 32)
    // Defaults from the Phase 5 sweep (results/sweep_bfs_balanced.csv).
    uint32_t smallMax = 16;     // gpu_binned: degree <  smallMax -> thread per vertex
    uint32_t largeMin = 256;    // gpu_binned: degree >= largeMin -> threadgroup per vertex
    uint32_t ept = 2;           // gpu_edge: edges per thread
};

class GpuBfs {
public:
    GpuBfs(MetalContext& ctx, const graph::CSR& g);

    const graph::CSR& graph() const { return view_; }
    uint32_t* dist() const { return static_cast<uint32_t*>(dist_.contents); }

    // gpu_vertex: thread per vertex every level.
    void runVertex(uint32_t source, uint32_t* out, const GpuBfsOptions& opt,
                   std::vector<util::BfsLevel>* trace = nullptr);
    // gpu_naive: thread per frontier vertex, CAS claim, atomic queue append.
    void runNaive(uint32_t source, uint32_t* out, const GpuBfsOptions& opt,
                  std::vector<util::BfsLevel>* trace = nullptr);
    // gpu_binned: frontier split by degree into thread / simdgroup /
    // threadgroup-per-vertex queues at discovery time; 3 kernels per level.
    void runBinned(uint32_t source, uint32_t* out, const GpuBfsOptions& opt,
                   std::vector<util::BfsLevel>* trace = nullptr);
    // gpu_edge: scan of frontier degrees, then a fixed number of edges per thread.
    void runEdge(uint32_t source, uint32_t* out, const GpuBfsOptions& opt,
                 std::vector<util::BfsLevel>* trace = nullptr);

    // Kernel-only time of the last run: sum of GPUEndTime - GPUStartTime over
    // its command buffers (excludes the host round trip between levels).
    double lastGpuMs() const { return lastGpuMs_; }
    // Number of levels (command buffers) of the last run.
    uint32_t lastLevels() const { return lastLevels_; }

private:
    void finish(uint32_t* out);
    // Start a level's command buffer; level 0 also gets the dist init kernel.
    id<MTLCommandBuffer> beginLevel(uint32_t level, uint32_t source,
                                    id<MTLComputeCommandEncoder>* enc);
    // Commit, wait, check, accumulate GPU time.
    void endLevel(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc);
    static uint32_t checkTg(uint32_t tg);

    MetalContext& ctx_;
    graph::CSR view_;
    uint32_t n_;
    id<MTLBuffer> rowPtr_, colIdx_, dist_, stats_;
    id<MTLBuffer> queueA_, queueB_;  // gpu_naive / gpu_edge frontiers, allocated on first use
    id<MTLBuffer> degA_, degB_;      // gpu_edge: degrees next to the queue entries (scanned in place)
    id<MTLBuffer> bins_[2][3];       // gpu_binned: [current/next][small/medium/large]
    id<MTLComputePipelineState> init_, vertexLevel_, naiveLevel_;
    id<MTLComputePipelineState> binSmall_, binMedium_, binLarge_, edgeExpand_;
    std::unique_ptr<Primitives> prims_;
    double lastGpuMs_ = 0;
    uint32_t lastLevels_ = 0;
};

} // namespace gpu
