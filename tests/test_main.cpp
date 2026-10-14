// gtest: tiny assert-based test runner (no googletest dependency).
// Register tests with TEST(name) { ... } and check with CHECK(cond).
// A failing CHECK records the failure and continues with the next test.
#include "metal/context.hpp"

#include <omp.h>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace {

struct TestCase { std::string name; std::function<void()> fn; };
std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; }
int g_failures = 0;
bool g_current_failed = false;

struct Registrar {
    Registrar(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

} // namespace

#define TEST(name)                                              \
    static void test_##name();                                  \
    static Registrar reg_##name(#name, test_##name);            \
    static void test_##name()

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::printf("    CHECK failed: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
            g_current_failed = true;                                                  \
            return;                                                                   \
        }                                                                             \
    } while (0)

// Shared GPU context: compiling shaders per test would dominate test time.
static gpu::MetalContext& ctx() { static gpu::MetalContext c; return c; }

// ---------------------------------------------------------------- Phase 0

TEST(openmp_has_threads) {
    int team = 0;
#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    CHECK(team >= 1);
}

// Vector add at awkward sizes: 0, 1, and non-multiples of the threadgroup size.
TEST(metal_vector_add_sizes) {
    id<MTLComputePipelineState> pso = ctx().pipeline("micro", "vector_add");
    for (uint32_t n : {0u, 1u, 255u, 256u, 257u, 100003u}) {
        std::vector<float> a(n), b(n);
        for (uint32_t i = 0; i < n; ++i) { a[i] = (float)i; b[i] = 2.0f * i + 1; }
        id<MTLBuffer> A = ctx().buffer(a.data(), n * 4), B = ctx().buffer(b.data(), n * 4);
        id<MTLBuffer> C = ctx().buffer(n * 4);
        id<MTLCommandBuffer> cb = [ctx().queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:B offset:0 atIndex:1];
        [enc setBuffer:C offset:0 atIndex:2];
        [enc setBytes:&n length:4 atIndex:3];
        gpu::MetalContext::dispatch1D(enc, pso, n, 256);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float* c = (const float*)C.contents;
        for (uint32_t i = 0; i < n; ++i) CHECK(c[i] == a[i] + b[i]);
    }
}

// ---------------------------------------------------------------- runner

int main() {
    int passed = 0;
    for (auto& t : registry()) {
        g_current_failed = false;
        std::printf("[ RUN  ] %s\n", t.name.c_str());
        t.fn();
        if (g_current_failed) { ++g_failures; std::printf("[ FAIL ] %s\n", t.name.c_str()); }
        else { ++passed; std::printf("[  OK  ] %s\n", t.name.c_str()); }
    }
    std::printf("\n%d passed, %d failed\n", passed, g_failures);
    return g_failures ? 1 : 0;
}
