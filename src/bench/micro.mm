// gmicro: microbenchmarks and smoke tests.
//   --hello   runtime-compile a vector-add kernel, run on 10M floats, verify,
//             print device properties
//   --omp     OpenMP hello world: report thread count and per-thread ids
//   --bandwidth  sequential read/write/copy GB/s, GPU and CPU, 1 MB -> 2 GB
//   --launch     kernel dispatch latency (sync per dispatch vs batched)
//   --random     random gather vs sequential, GPU and CPU, 1 MB -> 2 GB table
// Phase 3 results go to results/micro_{bandwidth,launch,random}.csv.
#include "metal/context.hpp"
#include "util/args.hpp"
#include "util/rng.hpp"
#include "util/sysinfo.hpp"
#include "util/timer.hpp"

#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
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

// ---------------------------------------------------------------------------
// Phase 3 microbenchmarks: shared helpers.
namespace {

constexpr size_t kMiB = size_t(1) << 20;
constexpr size_t kTG = 256;  // threadgroup size for every microbenchmark kernel

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "gmicro: %s\n", msg.c_str());
    std::exit(1);
}

// "timestamp,git_commit,chip": the first three columns of every results row.
std::string provenance() {
    return util::timestamp() + "," + util::gitCommit() + "," + util::chipName();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))];
}

// Untimed warm-up: repeat `runOnce` until at least `minMs` of wall time has
// passed (and at least once). One short warm-up run is not enough: the GPU
// ramps its clock over tens of milliseconds, and without this the first
// timed trials at small sizes ran up to 2x slower than the later ones.
template <class Run>
void warmUp(Run&& runOnce, double minMs = 200.0) {
    util::Timer t;
    do { runOnce(); } while (t.ms() < minMs);
}

// 1 MB, 2 MB, ... up to maxB (powers of two).
std::vector<size_t> sizeList(size_t minB, size_t maxB) {
    std::vector<size_t> v;
    for (size_t s = minB; s <= maxB; s *= 2) v.push_back(s);
    return v;
}

std::string humanBytes(size_t b) {
    char buf[32];
    if (b >= 1024 * kMiB) std::snprintf(buf, sizeof buf, "%zu GB", b / (1024 * kMiB));
    else std::snprintf(buf, sizeof buf, "%zu MB", b / kMiB);
    return buf;
}

// Encode `reps` back-to-back dispatches in ONE command buffer and return its
// GPU time in ms. Batching amortizes the ~100+ us command-buffer round trip,
// which would otherwise dominate small buffer sizes (that overhead is what
// --launch measures separately). The default serial dispatch type runs the
// dispatches one after another, so per-rep time = total / reps.
template <class Encode>
double gpuBatchMs(gpu::MetalContext& ctx, int reps, Encode encode) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (int r = 0; r < reps; ++r) encode(enc);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) die("command buffer failed");
        return gpu::MetalContext::gpuTimeMs(cb);
    }
}

// Sum of the per-threadgroup partials written by bw_read / gather_sum.
uint32_t sumPartials(id<MTLBuffer> partial, size_t groups) {
    const uint32_t* p = static_cast<const uint32_t*>(partial.contents);
    uint32_t s = 0;
    for (size_t i = 0; i < groups; ++i) s += p[i];
    return s;
}

// Grid for a grid-stride kernel over `items` elements: at most `maxThreads`,
// never more threads than items (rounded up to whole threadgroups).
size_t gridThreads(size_t items, size_t maxThreads) {
    size_t t = std::min(items, maxThreads);
    return (t + kTG - 1) / kTG * kTG;
}

// --- CPU (OpenMP) equivalents. schedule(static): equal contiguous chunks,
// which is optimal for uniform streaming work and keeps each thread on its
// own pages.

void cpuWrite(uint32_t* a, size_t n4, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < n4; ++i) {
        uint32_t v = (uint32_t)i;
        a[4 * i] = v; a[4 * i + 1] = v; a[4 * i + 2] = v; a[4 * i + 3] = v;
    }
}

// A plain `s += a[i]` reduction compiles to one scalar add per element
// (~12 GB/s on one core: compute-bound). Sixteen independent lanes let clang
// emit four 128-bit NEON adds per 64 bytes, so the loop is memory-bound.
uint32_t cpuRead(const uint32_t* a, size_t n, int threads) {
    uint32_t total = 0;
#pragma omp parallel num_threads(threads) reduction(+ : total)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t blocks = n / 16, per = (blocks + nt - 1) / nt;
        size_t lo = std::min(blocks, t * per) * 16, hi = std::min(blocks, (t + 1) * per) * 16;
        uint32_t lane[16] = {0};
        for (size_t i = lo; i < hi; i += 16)
            for (int k = 0; k < 16; ++k) lane[k] += a[i + k];
        for (int k = 0; k < 16; ++k) total += lane[k];
        if (t == nt - 1)
            for (size_t i = blocks * 16; i < n; ++i) total += a[i];  // tail
    }
    return total;
}

void cpuCopy(const uint8_t* a, uint8_t* b, size_t bytes, int threads) {
#pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t chunk = (bytes / nt + 63) & ~size_t(63);  // cache-line aligned chunks
        size_t lo = std::min(bytes, t * chunk), hi = std::min(bytes, lo + chunk);
        if (t == nt - 1) hi = bytes;
        std::memcpy(b + lo, a + lo, hi - lo);
    }
}

uint32_t cpuGather(const uint32_t* table, const uint32_t* idx, size_t m, int threads) {
    uint32_t s = 0;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(+ : s)
    for (size_t i = 0; i < m; ++i) s += table[idx[i]];
    return s;
}

// Expected value of bw_read / cpuRead after bw_write: sum over i of 4*i, mod 2^32.
uint32_t expectedWriteSum(size_t n4) {
    uint64_t n = n4;
    return (uint32_t)(4 * (n * (n - 1) / 2));  // n < 2^28, so no 64-bit overflow
}

bool checkWritePattern(const uint32_t* a, size_t n4, int threads) {
    size_t bad = 0;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(+ : bad)
    for (size_t i = 0; i < n4; ++i)
        for (int k = 0; k < 4; ++k) bad += a[4 * i + k] != (uint32_t)i;
    return bad == 0;
}

} // namespace

// ---------------------------------------------------------------------------
// --bandwidth: sequential read / write / copy, GPU vs CPU, over buffer sizes.
//
// Size = bytes per buffer. Traffic counted: read S, write S, copy 2S (read
// source + write destination). Copy needs two buffers, so it stops at
// --copy-max-mb (default 1 GB, i.e. 2 GB resident) to respect the 3 GB
// process memory limit; read and write go to the full 2 GB.
static int runBandwidth(const util::Args& args) {
    const size_t minB = (size_t)args.getInt("min-mb", 1) * kMiB;
    const size_t maxB = (size_t)args.getInt("max-mb", 2048) * kMiB;
    const size_t copyMax = (size_t)args.getInt("copy-max-mb", 1024) * kMiB;
    const int trials = (int)args.getInt("trials", 5);
    const int threads = (int)args.getInt("threads", omp_get_max_threads());
    const size_t maxGrid = (size_t)args.getInt("grid", 16777216);
    const size_t target = (size_t)args.getInt("target-mb", 1024) * kMiB;  // bytes per trial
    const std::string out = args.get("out", "results/micro_bandwidth.csv");
    const bool doGpu = !args.has("cpu-only"), doCpu = !args.has("gpu-only");

    gpu::MetalContext ctx;
    id<MTLComputePipelineState> psoRead = ctx.pipeline("micro", "bw_read");
    id<MTLComputePipelineState> psoWrite = ctx.pipeline("micro", "bw_write");
    id<MTLComputePipelineState> psoCopy = ctx.pipeline("micro", "bw_copy");

    std::ofstream csv(out, std::ios::trunc);
    csv << "timestamp,git_commit,chip,device,op,threads,bytes,reps,trial,time_ms,gbps,load1\n";
    const std::string prov = provenance();

    std::printf("sequential bandwidth, median GB/s over %d trials (GPU grid <= %zu threads, CPU %d threads)\n",
                trials, maxGrid, threads);
    std::printf("%8s | %8s %8s %8s | %8s %8s %8s\n", "size", "GPU rd", "GPU wr", "GPU cp",
                "CPU rd", "CPU wr", "CPU cp");
    bool allOk = true;

    for (size_t S : sizeList(minB, maxB)) {
        @autoreleasepool {
            const uint32_t n4 = (uint32_t)(S / 16);
            const bool copy = S <= copyMax;
            // Enough reps per trial to move ~target bytes: small sizes get
            // many reps (so timer resolution does not matter) and stay
            // resident in the GPU's caches between reps, which is the point.
            const int reps = (int)std::max<size_t>(1, std::min<size_t>(1024, target / S));
            const size_t grid = gridThreads(n4, maxGrid);
            const size_t groups = grid / kTG;

            id<MTLBuffer> A = ctx.buffer(S);
            id<MTLBuffer> B = copy ? ctx.buffer(S) : nil;
            id<MTLBuffer> partial = ctx.buffer(groups * sizeof(uint32_t));
            uint32_t* a = static_cast<uint32_t*>(A.contents);
            const uint32_t expect = expectedWriteSum(n4);

            double med[6] = {0, 0, 0, 0, 0, 0};
            auto record = [&](const char* dev, const char* op, int thr, size_t traffic,
                              const std::vector<double>& ms, int slot) {
                std::vector<double> g;
                for (size_t t = 0; t < ms.size(); ++t) {
                    double gbps = (double)traffic * reps / (ms[t] * 1e6);
                    g.push_back(gbps);
                    char row[256];
                    std::snprintf(row, sizeof row, ",%s,%s,%d,%zu,%d,%zu,%.4f,%.2f,%.2f\n", dev, op,
                                  thr, S, reps, t, ms[t], gbps, util::loadAverage1());
                    csv << prov << row;
                }
                med[slot] = median(g);
            };
            // Warm-up (first touch of fresh pages, GPU clock ramp), then timed trials.
            auto trialsOf = [&](auto&& runOnce) {
                std::vector<double> ms;
                warmUp(runOnce);
                for (int t = 0; t < trials; ++t) ms.push_back(runOnce());
                return ms;
            };

            if (doGpu) {
                auto wr = trialsOf([&] {
                    return gpuBatchMs(ctx, reps, [&](id<MTLComputeCommandEncoder> enc) {
                        [enc setBuffer:A offset:0 atIndex:0];
                        [enc setBytes:&n4 length:sizeof n4 atIndex:1];
                        gpu::MetalContext::dispatch1D(enc, psoWrite, grid, kTG);
                    });
                });
                if (!checkWritePattern(a, n4, threads)) { allOk = false; std::fprintf(stderr, "GPU write check FAILED at %zu\n", S); }
                record("gpu", "write", (int)grid, S, wr, 1);

                auto rd = trialsOf([&] {
                    return gpuBatchMs(ctx, reps, [&](id<MTLComputeCommandEncoder> enc) {
                        [enc setBuffer:A offset:0 atIndex:0];
                        [enc setBuffer:partial offset:0 atIndex:1];
                        [enc setBytes:&n4 length:sizeof n4 atIndex:2];
                        gpu::MetalContext::dispatch1D(enc, psoRead, grid, kTG);
                    });
                });
                if (sumPartials(partial, groups) != expect) { allOk = false; std::fprintf(stderr, "GPU read check FAILED at %zu\n", S); }
                record("gpu", "read", (int)grid, S, rd, 0);

                if (copy) {
                    std::memset(B.contents, 0, S);
                    auto cp = trialsOf([&] {
                        return gpuBatchMs(ctx, reps, [&](id<MTLComputeCommandEncoder> enc) {
                            [enc setBuffer:A offset:0 atIndex:0];
                            [enc setBuffer:B offset:0 atIndex:1];
                            [enc setBytes:&n4 length:sizeof n4 atIndex:2];
                            gpu::MetalContext::dispatch1D(enc, psoCopy, grid, kTG);
                        });
                    });
                    if (std::memcmp(A.contents, B.contents, S) != 0) { allOk = false; std::fprintf(stderr, "GPU copy check FAILED at %zu\n", S); }
                    record("gpu", "copy", (int)grid, 2 * S, cp, 2);
                }
            }

            if (doCpu) {
                // CPU and GPU share the same buffers (unified memory): no copies.
                auto wr = trialsOf([&] {
                    util::Timer t;
                    for (int r = 0; r < reps; ++r) cpuWrite(a, n4, threads);
                    return t.ms();
                });
                if (!checkWritePattern(a, n4, threads)) { allOk = false; std::fprintf(stderr, "CPU write check FAILED at %zu\n", S); }
                record("cpu", "write", threads, S, wr, 4);

                uint32_t got = 0;
                auto rd = trialsOf([&] {
                    util::Timer t;
                    for (int r = 0; r < reps; ++r) got = cpuRead(a, (size_t)n4 * 4, threads);
                    return t.ms();
                });
                if (got != expect) { allOk = false; std::fprintf(stderr, "CPU read check FAILED at %zu\n", S); }
                record("cpu", "read", threads, S, rd, 3);

                if (copy) {
                    std::memset(B.contents, 0, S);
                    auto cp = trialsOf([&] {
                        util::Timer t;
                        for (int r = 0; r < reps; ++r)
                            cpuCopy(static_cast<const uint8_t*>(A.contents),
                                    static_cast<uint8_t*>(B.contents), S, threads);
                        return t.ms();
                    });
                    if (std::memcmp(A.contents, B.contents, S) != 0) { allOk = false; std::fprintf(stderr, "CPU copy check FAILED at %zu\n", S); }
                    record("cpu", "copy", threads, 2 * S, cp, 5);
                }
            }

            auto cell = [](double v) {
                static char b[6][16];
                static int k = 0;
                char* p = b[k++ % 6];
                if (v > 0) std::snprintf(p, 16, "%8.1f", v); else std::snprintf(p, 16, "%8s", "-");
                return p;
            };
            std::printf("%8s | %s %s %s | %s %s %s\n", humanBytes(S).c_str(), cell(med[0]),
                        cell(med[1]), cell(med[2]), cell(med[3]), cell(med[4]), cell(med[5]));
            std::fflush(stdout);
        }
    }
    std::printf("verification: %s\nwrote %s\n", allOk ? "OK" : "FAILED", out.c_str());
    return allOk ? 0 : 1;
}

// ---------------------------------------------------------------------------
// --launch: what does one kernel dispatch cost?
//   sync_each     one command buffer per dispatch, commit + waitUntilCompleted
//                 each time: the full CPU->GPU->CPU round trip a level-
//                 synchronous BFS pays per level when the host must read a
//                 result (e.g. the next frontier size) before continuing.
//   async_each    one command buffer per dispatch, all committed back to
//                 back, wait only on the last: submission cost without the
//                 round trip.
//   encoder_each  one command buffer, a new compute encoder per dispatch.
//   one_encoder   one command buffer, one encoder, all dispatches: the
//                 floor (GPU-side cost between dependent dispatches).
static int runLaunch(const util::Args& args) {
    const int n = (int)args.getInt("dispatches", 1000);
    const int trials = (int)args.getInt("trials", 5);
    const std::string out = args.get("out", "results/micro_launch.csv");

    gpu::MetalContext ctx;
    id<MTLComputePipelineState> pso = ctx.pipeline("micro", "empty_kernel");
    id<MTLCommandQueue> q = ctx.queue();

    std::ofstream csv(out, std::ios::trunc);
    csv << "timestamp,git_commit,chip,mode,dispatches,trial,wall_ms,gpu_ms,"
           "wall_us_per_dispatch,gpu_us_per_dispatch,median_wall_us,p90_wall_us,load1\n";
    const std::string prov = provenance();

    auto encodeOne = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    };
    auto check = [](id<MTLCommandBuffer> cb) {
        if (cb.status != MTLCommandBufferStatusCompleted) die("command buffer failed");
    };

    struct Result { double wallMs, gpuMs, medUs, p90Us; };
    auto syncEach = [&]() {
        std::vector<double> us;
        double gpu = 0;
        util::Timer total;
        for (int i = 0; i < n; ++i) {
            @autoreleasepool {
                util::Timer t;
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                encodeOne(enc);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                us.push_back(t.ms() * 1e3);
                check(cb);
                gpu += gpu::MetalContext::gpuTimeMs(cb);
            }
        }
        return Result{total.ms(), gpu, median(us), percentile(us, 0.9)};
    };
    auto asyncEach = [&]() {
        @autoreleasepool {
            util::Timer total;
            id<MTLCommandBuffer> first = nil, last = nil;
            for (int i = 0; i < n; ++i) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                encodeOne(enc);
                [enc endEncoding];
                [cb commit];
                if (!first) first = cb;
                last = cb;
            }
            [last waitUntilCompleted];  // a queue executes its command buffers in order
            double wall = total.ms();
            check(last);
            return Result{wall, (last.GPUEndTime - first.GPUStartTime) * 1e3, -1, -1};
        }
    };
    auto oneBuffer = [&](bool encoderEach) {
        @autoreleasepool {
            util::Timer total;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = nil;
            for (int i = 0; i < n; ++i) {
                if (!enc) enc = [cb computeCommandEncoder];
                encodeOne(enc);
                if (encoderEach) { [enc endEncoding]; enc = nil; }
            }
            if (enc) [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double wall = total.ms();
            check(cb);
            return Result{wall, gpu::MetalContext::gpuTimeMs(cb), -1, -1};
        }
    };

    // Warm-up: pipeline creation done above; a few hundred round trips ramp
    // the GPU out of its idle state before anything is timed.
    for (int i = 0; i < 3; ++i) syncEach();

    struct Mode { const char* name; std::function<Result()> run; };
    std::vector<Mode> modes = {
        {"sync_each", syncEach},
        {"async_each", asyncEach},
        {"encoder_each", [&] { return oneBuffer(true); }},
        {"one_encoder", [&] { return oneBuffer(false); }},
    };

    std::printf("kernel launch cost, %d empty dispatches (1 threadgroup of 32), median of %d trials\n",
                n, trials);
    std::printf("%-13s %14s %14s %12s %12s\n", "mode", "wall us/disp", "GPU us/disp", "median us",
                "p90 us");
    for (auto& m : modes) {
        std::vector<double> wall, gpu, med, p90;
        for (int t = 0; t < trials; ++t) {
            Result r = m.run();
            double wus = r.wallMs * 1e3 / n, gus = r.gpuMs * 1e3 / n;
            wall.push_back(wus); gpu.push_back(gus);
            char row[320], extra[64] = ",,";
            if (r.medUs >= 0) {
                std::snprintf(extra, sizeof extra, ",%.2f,%.2f", r.medUs, r.p90Us);
                med.push_back(r.medUs); p90.push_back(r.p90Us);
            }
            std::snprintf(row, sizeof row, ",%s,%d,%d,%.4f,%.4f,%.3f,%.3f%s,%.2f\n", m.name, n, t,
                          r.wallMs, r.gpuMs, wus, gus, extra, util::loadAverage1());
            csv << prov << row;
        }
        if (med.empty())
            std::printf("%-13s %14.2f %14.2f %12s %12s\n", m.name, median(wall), median(gpu), "-", "-");
        else
            std::printf("%-13s %14.2f %14.2f %12.2f %12.2f\n", m.name, median(wall), median(gpu),
                        median(med), median(p90));
    }
    std::printf("wrote %s\n", out.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// --random: gather table[idx[i]] with random vs sequential idx, GPU vs CPU.
//
// The table size sweeps 1 MB -> 2 GB; the number of reads is fixed (--reads,
// default 32M, a 128 MB index array) so every point does the same work.
// Sequential: idx[i] = i mod table_len. Random: uniform in [0, table_len).
// Reported rates:
//   gbps_payload  4 B per gathered element / time (useful bytes)
//   gbps_total    payload + the 4 B idx element streamed per read
//   greads_per_s  billions of gathered elements per second
// A random 4 B read pulls a whole cache line from DRAM, so for large tables
// the actual DRAM traffic is much higher than gbps_total.
static int runRandom(const util::Args& args) {
    const size_t minB = (size_t)args.getInt("min-mb", 1) * kMiB;
    const size_t maxB = (size_t)args.getInt("max-mb", 2048) * kMiB;
    const size_t m = (size_t)args.getInt("reads", 32 * 1024 * 1024);
    const int trials = (int)args.getInt("trials", 5);
    const int threads = (int)args.getInt("threads", omp_get_max_threads());
    const size_t maxGrid = (size_t)args.getInt("grid", 16777216);
    // Allocate at least this much for the table (only S bytes are used):
    // separates "table size" from "allocation size" effects.
    const size_t minAlloc = (size_t)args.getInt("min-alloc-mb", 0) * kMiB;
    const std::string out = args.get("out", "results/micro_random.csv");
    const bool doGpu = !args.has("cpu-only"), doCpu = !args.has("gpu-only");

    gpu::MetalContext ctx;
    id<MTLComputePipelineState> pso = ctx.pipeline("micro", "gather_sum");

    std::ofstream csv(out, std::ios::trunc);
    csv << "timestamp,git_commit,chip,device,pattern,threads,table_bytes,reads,trial,time_ms,"
           "gbps_payload,gbps_total,greads_per_s,load1\n";
    const std::string prov = provenance();

    const uint32_t m32 = (uint32_t)m;
    const size_t grid = gridThreads(m, maxGrid), groups = grid / kTG;
    id<MTLBuffer> idxBuf = ctx.buffer(m * sizeof(uint32_t));
    id<MTLBuffer> partial = ctx.buffer(groups * sizeof(uint32_t));
    uint32_t* idx = static_cast<uint32_t*>(idxBuf.contents);

    std::printf("gather bandwidth, %zu reads (idx array %zu MB), median Greads/s over %d trials\n",
                m, m * 4 / kMiB, trials);
    std::printf("%8s | %9s %9s %7s | %9s %9s %7s\n", "table", "GPU seq", "GPU rand", "ratio",
                "CPU seq", "CPU rand", "ratio");
    bool allOk = true;

    for (size_t S : sizeList(minB, maxB)) {
        @autoreleasepool {
            const size_t len = S / sizeof(uint32_t);
            id<MTLBuffer> T = ctx.buffer(std::max(S, minAlloc));
            uint32_t* table = static_cast<uint32_t*>(T.contents);
#pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t i = 0; i < len; ++i) table[i] = (uint32_t)(i * 2654435761u);

            double med[4] = {0, 0, 0, 0};  // GPU seq, GPU rand, CPU seq, CPU rand
            for (int p = 0; p < 2; ++p) {
                const bool random = p == 1;
                const char* pattern = random ? "random" : "seq";
                // Fill idx in fixed 1M-element chunks, each with its own seeded
                // RNG: deterministic for any thread count, and parallel.
                const size_t chunk = size_t(1) << 20;
#pragma omp parallel for num_threads(threads) schedule(static)
                for (size_t c = 0; c < (m + chunk - 1) / chunk; ++c) {
                    util::Rng rng(0x5EEDull ^ (S * 31 + c));
                    size_t hi = std::min(m, (c + 1) * chunk);
                    for (size_t i = c * chunk; i < hi; ++i)
                        idx[i] = random ? (uint32_t)rng.below(len) : (uint32_t)(i % len);
                }
                // Serial reference sum: the GPU and CPU results must equal it.
                uint32_t expect = 0;
                for (size_t i = 0; i < m; ++i) expect += table[idx[i]];

                auto record = [&](const char* dev, int thr, const std::vector<double>& ms, int slot) {
                    std::vector<double> rate;
                    for (size_t t = 0; t < ms.size(); ++t) {
                        double payload = 4.0 * m / (ms[t] * 1e6), total = 2 * payload;
                        double greads = m / (ms[t] * 1e6);
                        rate.push_back(greads);
                        char row[256];
                        std::snprintf(row, sizeof row, ",%s,%s,%d,%zu,%zu,%zu,%.4f,%.2f,%.2f,%.4f,%.2f\n",
                                      dev, pattern, thr, S, m, t, ms[t], payload, total, greads,
                                      util::loadAverage1());
                        csv << prov << row;
                    }
                    med[slot] = median(rate);
                };

                if (doGpu) {
                    auto once = [&] {
                        return gpuBatchMs(ctx, 1, [&](id<MTLComputeCommandEncoder> enc) {
                            [enc setBuffer:T offset:0 atIndex:0];
                            [enc setBuffer:idxBuf offset:0 atIndex:1];
                            [enc setBuffer:partial offset:0 atIndex:2];
                            [enc setBytes:&m32 length:sizeof m32 atIndex:3];
                            gpu::MetalContext::dispatch1D(enc, pso, grid, kTG);
                        });
                    };
                    std::vector<double> ms;
                    warmUp(once);
                    for (int t = 0; t < trials; ++t) ms.push_back(once());
                    if (sumPartials(partial, groups) != expect) {
                        allOk = false;
                        std::fprintf(stderr, "GPU gather check FAILED (%s, %zu)\n", pattern, S);
                    }
                    record("gpu", (int)grid, ms, random ? 1 : 0);
                }
                if (doCpu) {
                    uint32_t got = 0;
                    std::vector<double> ms;
                    warmUp([&] { got = cpuGather(table, idx, m, threads); });
                    for (int t = 0; t < trials; ++t) {
                        util::Timer tm;
                        got = cpuGather(table, idx, m, threads);
                        ms.push_back(tm.ms());
                    }
                    if (got != expect) {
                        allOk = false;
                        std::fprintf(stderr, "CPU gather check FAILED (%s, %zu)\n", pattern, S);
                    }
                    record("cpu", threads, ms, random ? 3 : 2);
                }
            }
            auto half = [](double sq, double rnd) {  // "  seq  rand  ratio" or dashes if skipped
                char b[48];
                if (sq > 0 && rnd > 0) std::snprintf(b, sizeof b, "%9.3f %9.3f %6.1fx", sq, rnd, sq / rnd);
                else std::snprintf(b, sizeof b, "%9s %9s %7s", "-", "-", "-");
                return std::string(b);
            };
            std::printf("%8s | %s | %s\n", humanBytes(S).c_str(), half(med[0], med[1]).c_str(),
                        half(med[2], med[3]).c_str());
            std::fflush(stdout);
        }
    }
    std::printf("verification: %s\nwrote %s\n", allOk ? "OK" : "FAILED", out.c_str());
    return allOk ? 0 : 1;
}

int main(int argc, char** argv) {
    util::Args args(argc, argv);
    if (args.has("hello")) return runHello();
    if (args.has("omp")) return runOmp();
    if (args.has("bandwidth")) return runBandwidth(args);
    if (args.has("launch")) return runLaunch(args);
    if (args.has("random")) return runRandom(args);
    std::fprintf(stderr,
                 "usage: gmicro --hello | --omp\n"
                 "       gmicro --bandwidth [--min-mb 1] [--max-mb 2048] [--copy-max-mb 1024]\n"
                 "                          [--trials 5] [--threads N] [--grid 16777216] [--target-mb 1024]\n"
                 "                          [--gpu-only|--cpu-only] [--out FILE]\n"
                 "       gmicro --launch    [--dispatches 1000] [--trials 5] [--out FILE]\n"
                 "       gmicro --random    [--min-mb 1] [--max-mb 2048] [--reads 33554432]\n"
                 "                          [--trials 5] [--threads N] [--grid 16777216] [--min-alloc-mb 0]\n"
                 "                          [--gpu-only|--cpu-only] [--out FILE]\n");
    return 2;
}
