#pragma once
// MetalContext: the single entry point to the GPU.
//   - device + command queue
//   - runtime shader compilation (common.h + <file>.metal), cached per file
//   - compute pipeline cache keyed by kernel function name
//   - shared-storage buffer helpers and a 1D dispatch helper
//   - GPU-side timing from the command buffer
//
// Objective-C++ only: include from .mm files (or files compiled as ObjC++).
#ifndef __OBJC__
#error "context.hpp requires Objective-C++ (compile as .mm or with -x objective-c++)"
#endif

#import <Metal/Metal.h>
#include <cstdio>
#include <string>
#include <unordered_map>

namespace gpu {

class MetalContext {
public:
    MetalContext();

    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }

    // Print device properties (name, limits, families) to `out`.
    void printDeviceInfo(FILE* out = stdout) const;

    // Pipeline for kernel `fn` defined in src/shaders/<file>.metal.
    // Compiles the file on first use; aborts with the compiler log on failure.
    id<MTLComputePipelineState> pipeline(const std::string& file, const std::string& fn);

    // All buffers use shared storage: CPU and GPU see the same bytes (unified
    // memory), so results can be read directly after waitUntilCompleted.
    id<MTLBuffer> buffer(size_t bytes);
    id<MTLBuffer> buffer(const void* data, size_t bytes);

    // Encode a 1D dispatch of `n` threads in threadgroups of `tg` threads.
    // Uses dispatchThreadgroups (not dispatchThreads), so the last group may
    // run past n: every kernel must bounds-check against its element count.
    static void dispatch1D(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso, size_t n, size_t tg = 256);

    // Kernel-only time of a completed command buffer, in milliseconds.
    static double gpuTimeMs(id<MTLCommandBuffer> cb) {
        return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    }

private:
    id<MTLLibrary> library(const std::string& file);

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    std::string shaderDir_;
    std::unordered_map<std::string, id<MTLLibrary>> libs_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> psos_;
};

// Directory holding the .metal files: $GGPU_SHADER_DIR, else the build-time path.
std::string shaderDir();

} // namespace gpu
