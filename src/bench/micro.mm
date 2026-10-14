// gmicro: microbenchmarks and smoke tests.
//   --hello   runtime-compile a vector-add kernel, run on 10M floats, verify,
//             print device properties
//   --omp     OpenMP hello world: report thread count and per-thread ids
// (Phase 3 adds --bandwidth, --launch, --random.)
#include "metal/context.hpp"
#include "util/args.hpp"
#include "util/timer.hpp"

#include <omp.h>
#include <cstdio>
#include <vector>

static int runHello() {
    gpu::MetalContext ctx;
    ctx.printDeviceInfo();

    const uint32_t n = 10'000'000;
    std::vector<float> a(n), b(n);
    for (uint32_t i = 0; i < n; ++i) {
        a[i] = (float)(i % 1000) * 0.5f;
        b[i] = (float)(i % 777) * 0.25f;
    }

    util::Timer compileTimer;
    id<MTLComputePipelineState> pso = ctx.pipeline("micro", "vector_add");
    double compileMs = compileTimer.ms();
    std::printf("  threadExecutionWidth   : %lu (simdgroup size)\n",
                (unsigned long)pso.threadExecutionWidth);
    std::printf("  pipeline maxThreads/TG : %lu\n", (unsigned long)pso.maxTotalThreadsPerThreadgroup);

    id<MTLBuffer> bufA = ctx.buffer(a.data(), n * sizeof(float));
    id<MTLBuffer> bufB = ctx.buffer(b.data(), n * sizeof(float));
    id<MTLBuffer> bufC = ctx.buffer(n * sizeof(float));

    util::Timer e2e;
    id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    gpu::MetalContext::dispatch1D(enc, pso, n, 256);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    double e2eMs = e2e.ms();

    // Shared storage: read the GPU's output in place, no copy.
    const float* c = static_cast<const float*>(bufC.contents);
    size_t bad = 0;
    for (uint32_t i = 0; i < n; ++i)
        if (c[i] != a[i] + b[i]) ++bad;  // exact: one IEEE add on both sides

    double gpuMs = gpu::MetalContext::gpuTimeMs(cb);
    double gbps = 3.0 * n * sizeof(float) / (gpuMs * 1e6);
    std::printf("\nvector_add, n = %u floats\n", n);
    std::printf("  shader compile         : %.1f ms (runtime, includes common.h)\n", compileMs);
    std::printf("  GPU time               : %.3f ms (%.1f GB/s effective, first run)\n", gpuMs, gbps);
    std::printf("  end-to-end             : %.3f ms\n", e2eMs);
    std::printf("  verification           : %s (%zu mismatches)\n", bad ? "FAILED" : "OK", bad);
    return bad ? 1 : 0;
}

static int runOmp() {
    std::printf("omp_get_max_threads()    : %d\n", omp_get_max_threads());
    std::printf("omp_get_num_procs()      : %d\n", omp_get_num_procs());
    std::vector<int> seen(omp_get_max_threads(), 0);
    int team = 0;
#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();
        seen[omp_get_thread_num()] = 1;
    }
    int count = 0;
    for (int s : seen) count += s;
    std::printf("parallel region team size: %d (%d distinct thread ids ran)\n", team, count);
    return count == team ? 0 : 1;
}

int main(int argc, char** argv) {
    util::Args args(argc, argv);
    if (args.has("hello")) return runHello();
    if (args.has("omp")) return runOmp();
    std::fprintf(stderr, "usage: gmicro --hello | --omp\n");
    return 2;
}
