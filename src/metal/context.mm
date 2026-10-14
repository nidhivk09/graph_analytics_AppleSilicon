#include "metal/context.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace gpu {

[[noreturn]] static void die(const std::string& msg) {
    std::fprintf(stderr, "[metal] fatal: %s\n", msg.c_str());
    std::exit(1);
}

static std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) die("cannot open shader file " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string shaderDir() {
    if (const char* env = std::getenv("GGPU_SHADER_DIR")) return env;
#ifdef GGPU_SHADER_DIR
    return GGPU_SHADER_DIR;
#else
    return "src/shaders";
#endif
}

MetalContext::MetalContext() : shaderDir_(shaderDir()) {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) die("no Metal device");
    queue_ = [device_ newCommandQueue];
    if (!queue_) die("cannot create command queue");
}

void MetalContext::printDeviceInfo(FILE* out) const {
    MTLSize tg = device_.maxThreadsPerThreadgroup;
    auto mib = [](uint64_t b) { return (double)b / (1024.0 * 1024.0); };
    std::fprintf(out, "Metal device             : %s\n", device_.name.UTF8String);
    std::fprintf(out, "  unified memory         : %s\n", device_.hasUnifiedMemory ? "yes" : "no");
    std::fprintf(out, "  GPU family Apple8      : %s\n",
                 [device_ supportsFamily:MTLGPUFamilyApple8] ? "yes" : "no");
    std::fprintf(out, "  GPU family Metal3      : %s\n",
                 [device_ supportsFamily:MTLGPUFamilyMetal3] ? "yes" : "no");
    std::fprintf(out, "  maxThreadsPerThreadgroup: %lu x %lu x %lu\n",
                 (unsigned long)tg.width, (unsigned long)tg.height, (unsigned long)tg.depth);
    std::fprintf(out, "  maxThreadgroupMemory   : %lu bytes\n",
                 (unsigned long)device_.maxThreadgroupMemoryLength);
    std::fprintf(out, "  maxBufferLength        : %.0f MiB\n", mib(device_.maxBufferLength));
    std::fprintf(out, "  recommendedMaxWorkingSetSize: %.0f MiB\n",
                 mib(device_.recommendedMaxWorkingSetSize));
}

id<MTLLibrary> MetalContext::library(const std::string& file) {
    auto it = libs_.find(file);
    if (it != libs_.end()) return it->second;

    // Runtime compilation does not resolve local #includes, so prepend common.h.
    std::string src = readFile(shaderDir_ + "/common.h") + "\n" +
                      readFile(shaderDir_ + "/" + file + ".metal");

    MTLCompileOptions* opts = [MTLCompileOptions new];
    // Fast math off: reproducible float results (PageRank is compared to a
    // double-precision reference) and IEEE-correct division/NaN handling.
    opts.fastMathEnabled = NO;
    // MSL 3.0 is needed for atomic_float (PageRank push variant).
    opts.languageVersion = MTLLanguageVersion3_0;

    NSError* err = nil;
    id<MTLLibrary> lib = [device_ newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                               options:opts
                                                 error:&err];
    if (!lib) die("compiling " + file + ".metal failed:\n" + err.localizedDescription.UTF8String);
    libs_[file] = lib;
    return lib;
}

id<MTLComputePipelineState> MetalContext::pipeline(const std::string& file, const std::string& fn) {
    auto it = psos_.find(fn);
    if (it != psos_.end()) return it->second;

    id<MTLFunction> f = [library(file) newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]];
    if (!f) die("kernel '" + fn + "' not found in " + file + ".metal");

    NSError* err = nil;
    id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:f error:&err];
    if (!pso) die("pipeline for '" + fn + "' failed: " + err.localizedDescription.UTF8String);

    // Several kernels (simd_sum, simd_ballot, ...) assume 32-wide simdgroups.
    if (pso.threadExecutionWidth != 32)
        die("expected threadExecutionWidth 32, got " + std::to_string(pso.threadExecutionWidth));

    psos_[fn] = pso;
    return pso;
}

id<MTLBuffer> MetalContext::buffer(size_t bytes) {
    // Metal rejects zero-length buffers; size-0 inputs still need a valid binding.
    id<MTLBuffer> b = [device_ newBufferWithLength:(bytes ? bytes : 4)
                                           options:MTLResourceStorageModeShared];
    if (!b) die("buffer allocation of " + std::to_string(bytes) + " bytes failed");
    return b;
}

id<MTLBuffer> MetalContext::buffer(const void* data, size_t bytes) {
    if (bytes == 0) return buffer(0);
    id<MTLBuffer> b = [device_ newBufferWithBytes:data
                                           length:bytes
                                          options:MTLResourceStorageModeShared];
    if (!b) die("buffer allocation of " + std::to_string(bytes) + " bytes failed");
    return b;
}

void MetalContext::dispatch1D(id<MTLComputeCommandEncoder> enc,
                              id<MTLComputePipelineState> pso, size_t n, size_t tg) {
    if (n == 0) return;
    if (tg > pso.maxTotalThreadsPerThreadgroup) tg = pso.maxTotalThreadsPerThreadgroup;
    [enc setComputePipelineState:pso];
    [enc dispatchThreadgroups:MTLSizeMake((n + tg - 1) / tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

} // namespace gpu
