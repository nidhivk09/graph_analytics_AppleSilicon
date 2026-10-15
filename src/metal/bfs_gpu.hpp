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
#include "util/trace.hpp"

#include <vector>

namespace gpu {

struct GpuBfsOptions {
    uint32_t tg = 256;  // threadgroup size for the level kernels (Phase 10 sweeps it)
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

    // Kernel-only time of the last run: sum of GPUEndTime - GPUStartTime over
    // its command buffers (excludes the host round trip between levels).
    double lastGpuMs() const { return lastGpuMs_; }
    // Number of levels (command buffers) of the last run.
    uint32_t lastLevels() const { return lastLevels_; }

private:
    void finish(uint32_t* out);

    MetalContext& ctx_;
    graph::CSR view_;
    uint32_t n_;
    id<MTLBuffer> rowPtr_, colIdx_, dist_, stats_;
    id<MTLBuffer> queueA_, queueB_;  // gpu_naive frontiers, allocated on first use
    id<MTLComputePipelineState> init_, vertexLevel_, naiveLevel_;
    double lastGpuMs_ = 0;
    uint32_t lastLevels_ = 0;
};

} // namespace gpu
