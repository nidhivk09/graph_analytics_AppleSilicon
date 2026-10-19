#include "cpu/bfs_steps.hpp"

#include <omp.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace cpu::steps {

namespace {

constexpr uint32_t kInf = 0xFFFFFFFFu;

int resolve(int threads) { return threads > 0 ? threads : omp_get_max_threads(); }

// Claim u for `value` (only one caller wins). The plain-load pre-check skips
// the CAS for visited vertices, the common case.
inline bool claim(uint32_t* dist, uint32_t u, uint32_t value) {
    if (__atomic_load_n(&dist[u], __ATOMIC_RELAXED) != kInf) return false;
    uint32_t expected = kInf;
    return __atomic_compare_exchange_n(&dist[u], &expected, value, false, __ATOMIC_RELAXED,
                                       __ATOMIC_RELAXED);
}

} // namespace

void initDist(uint32_t* dist, uint64_t n, uint32_t source, int threads) {
#pragma omp parallel for num_threads(resolve(threads)) schedule(static) if (n > (1u << 16))
    for (uint64_t v = 0; v < n; ++v) dist[v] = kInf;
    dist[source] = 0;
}

StepOut topDown(const graph::CSR& g, uint32_t* dist, const uint32_t* q, uint64_t size,
                uint64_t edges, uint32_t* qNext, uint32_t* degNext, uint32_t level, int threads) {
    StepOut out;
    uint64_t tail = 0;
    const int nt = edges < kSerialEdges ? 1 : resolve(threads);
#pragma omp parallel num_threads(nt)
    {
        // Per-thread output, appended with one atomic per thread at the end.
        // thread_local: OpenMP reuses its threads, so the capacity persists
        // across levels and runs (no allocation in the common case).
        static thread_local std::vector<uint32_t> lq, ld;
        lq.clear();
        ld.clear();
        uint64_t myEdges = 0, myExamined = 0;
#pragma omp for schedule(dynamic, 64) nowait
        for (uint64_t i = 0; i < size; ++i) {
            uint32_t v = q[i];
            myExamined += g.degree(v);
            for (const uint32_t* p = g.begin(v); p != g.end(v); ++p)
                if (claim(dist, *p, level + 1)) {
                    uint32_t d = g.degree(*p);
                    lq.push_back(*p);
                    ld.push_back(d);
                    myEdges += d;
                }
        }
        uint64_t pos = __atomic_fetch_add(&tail, (uint64_t)lq.size(), __ATOMIC_RELAXED);
        if (!lq.empty()) {
            std::memcpy(qNext + pos, lq.data(), lq.size() * sizeof(uint32_t));
            std::memcpy(degNext + pos, ld.data(), ld.size() * sizeof(uint32_t));
        }
        __atomic_fetch_add(&out.nextEdges, myEdges, __ATOMIC_RELAXED);
        __atomic_fetch_add(&out.examined, myExamined, __ATOMIC_RELAXED);
    }
    out.found = tail;
    return out;
}

StepOut bottomUp(const graph::CSR& g, uint32_t* dist, const uint32_t* front, uint32_t* next,
                 uint32_t level, int threads) {
    const uint64_t n = g.n, words = (n + 31) / 32;
    uint64_t found = 0, examined = 0, nextEdges = 0;
    // A thread owns whole words, so it writes `next` without atomics, like the GPU kernel.
#pragma omp parallel for num_threads(resolve(threads)) schedule(dynamic, 64) \
    reduction(+ : found, examined, nextEdges)
    for (uint64_t w = 0; w < words; ++w) {
        uint32_t word = 0;
        const uint64_t hi = std::min(n, (w + 1) * 32);
        for (uint64_t v = w * 32; v < hi; ++v) {
            if (dist[v] != kInf) continue;
            for (const uint32_t* p = g.begin((uint32_t)v); p != g.end((uint32_t)v); ++p) {
                ++examined;
                if ((front[*p >> 5] >> (*p & 31)) & 1u) {
                    dist[v] = level + 1;
                    word |= 1u << (v & 31);
                    ++found;
                    nextEdges += g.degree((uint32_t)v);
                    break;
                }
            }
        }
        next[w] = word;
    }
    return {found, examined, nextEdges};
}

void queueToBitmap(const uint32_t* q, uint64_t size, uint32_t* bitmap, uint64_t n, int threads) {
    const uint64_t words = (n + 31) / 32;
    const int nt = size < kSerialEdges ? 1 : resolve(threads);
#pragma omp parallel num_threads(nt)
    {
#pragma omp for schedule(static)
        for (uint64_t w = 0; w < words; ++w) bitmap[w] = 0;
#pragma omp for schedule(static)
        for (uint64_t i = 0; i < size; ++i)
            __atomic_fetch_or(&bitmap[q[i] >> 5], 1u << (q[i] & 31), __ATOMIC_RELAXED);
    }
}

uint64_t bitmapToQueue(const graph::CSR& g, const uint32_t* bitmap, uint32_t* q, uint32_t* deg,
                       int threads) {
    const uint64_t words = (g.n + 31) / 32;
    const int nt = resolve(threads);
    std::vector<uint64_t> start(nt + 1, 0);
    int team = 1;  // the runtime may give fewer threads than requested
    // Two passes over static word ranges: count, prefix, write. Output is in
    // increasing vertex order.
#pragma omp parallel num_threads(nt)
    {
        const int t = omp_get_thread_num(), T = omp_get_num_threads();
        const uint64_t lo = words * t / T, hi = words * (t + 1) / T;
        uint64_t c = 0;
        for (uint64_t w = lo; w < hi; ++w) c += __builtin_popcount(bitmap[w]);
        start[t + 1] = c;
#pragma omp barrier
#pragma omp single
        {
            team = T;
            for (int i = 0; i < T; ++i) start[i + 1] += start[i];
        }
        uint64_t pos = start[t];
        for (uint64_t w = lo; w < hi; ++w)
            for (uint32_t bits = bitmap[w]; bits; bits &= bits - 1) {
                uint32_t v = (uint32_t)(w * 32 + __builtin_ctz(bits));
                q[pos] = v;
                deg[pos] = g.degree(v);
                ++pos;
            }
    }
    return start[team];
}

} // namespace cpu::steps
