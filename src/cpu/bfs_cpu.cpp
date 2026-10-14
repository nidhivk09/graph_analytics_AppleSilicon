#include "cpu/bfs_cpu.hpp"

#include <omp.h>
#include <cstring>
#include <stdexcept>

namespace cpu {

Schedule parseSchedule(const std::string& s) {
    if (s == "static") return Schedule::Static;
    if (s == "dynamic" || s == "dynamic,64" || s == "dynamic/64") return Schedule::Dynamic64;
    if (s == "guided") return Schedule::Guided;
    throw std::runtime_error("unknown schedule '" + s + "' (static | dynamic,64 | guided)");
}

const char* scheduleName(Schedule s) {
    switch (s) {
        case Schedule::Static: return "static";
        case Schedule::Dynamic64: return "dynamic/64";
        case Schedule::Guided: return "guided";
    }
    return "?";
}

// ------------------------------------------------------------------ serial

void bfsSerial(const graph::CSR& g, uint32_t source, uint32_t* dist) {
    std::fill(dist, dist + g.n, kUnreached);
    // A flat array as the FIFO: each vertex is enqueued at most once, so n
    // slots suffice and there is no deque bookkeeping.
    std::vector<uint32_t> queue(g.n);
    size_t head = 0, tail = 0;
    dist[source] = 0;
    queue[tail++] = source;
    while (head < tail) {
        uint32_t u = queue[head++];
        uint32_t du = dist[u] + 1;
        for (const uint32_t* p = g.begin(u); p != g.end(u); ++p) {
            if (dist[*p] == kUnreached) {
                dist[*p] = du;
                queue[tail++] = *p;
            }
        }
    }
}

std::vector<uint32_t> bfsSerial(const graph::CSR& g, uint32_t source) {
    std::vector<uint32_t> dist(g.n);
    bfsSerial(g, source, dist.data());
    return dist;
}

// ------------------------------------------------------------------ helpers

namespace {

int threadCount(const BfsOptions& o) { return o.threads > 0 ? o.threads : omp_get_max_threads(); }

void parallelFill(uint32_t* a, int64_t n, uint32_t v, int nt) {
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int64_t i = 0; i < n; ++i) a[i] = v;
}

// Claim v for distance d. The plain load first skips the (expensive) atomic
// for the common case of an already-visited neighbor. Relaxed ordering is
// enough: the only thing that must be agreed on is *who* wins the CAS, and
// the level barrier orders everything else.
inline bool claim(uint32_t* dist, uint32_t v, uint32_t d) {
    uint32_t cur = __atomic_load_n(&dist[v], __ATOMIC_RELAXED);
    if (cur != kUnreached) return false;
    return __atomic_compare_exchange_n(&dist[v], &cur, d, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}

struct StepResult {
    uint64_t next = 0;      // size of the next frontier
    uint64_t scout = 0;     // sum of degrees of newly discovered vertices (m_f)
    uint64_t examined = 0;  // adjacency entries looked at
};

// Threads collect discoveries in a private buffer and reserve space in the
// shared next-frontier queue in blocks: one atomic per kLocal vertices
// instead of one per vertex.
constexpr size_t kLocal = 4096;

// One top-down level. The loop schedule comes from omp_set_schedule
// (schedule(runtime)), which is how cpu_omp makes it configurable.
StepResult topDownStep(const graph::CSR& g, uint32_t* dist, const uint32_t* front, int64_t fsize,
                       uint32_t* next, uint32_t d, int nt) {
    uint64_t tail = 0, scout = 0, examined = 0;
#pragma omp parallel num_threads(nt) reduction(+ : scout, examined)
    {
        std::vector<uint32_t> local;
        local.reserve(kLocal);
        auto flush = [&] {
            if (local.empty()) return;
            uint64_t pos = __atomic_fetch_add(&tail, (uint64_t)local.size(), __ATOMIC_RELAXED);
            std::memcpy(next + pos, local.data(), local.size() * sizeof(uint32_t));
            local.clear();
        };
#pragma omp for schedule(runtime) nowait
        for (int64_t i = 0; i < fsize; ++i) {
            uint32_t u = front[i];
            examined += g.degree(u);
            for (const uint32_t* p = g.begin(u); p != g.end(u); ++p) {
                if (claim(dist, *p, d)) {
                    scout += g.degree(*p);
                    local.push_back(*p);
                    if (local.size() == kLocal) flush();
                }
            }
        }
        flush();
    }
    return {tail, scout, examined};
}

// Dense frontier: one bit per vertex. Bottom-up probes it for every
// neighbor, so it must be small enough to stay in cache (n/8 bytes:
// 600 KB for LiveJournal).
struct Bitmap {
    std::vector<uint64_t> w;
    explicit Bitmap(uint64_t n) : w((n + 63) / 64, 0) {}
    bool get(uint32_t i) const { return (w[i >> 6] >> (i & 63)) & 1; }
    void setAtomic(uint32_t i) { __atomic_fetch_or(&w[i >> 6], 1ull << (i & 63), __ATOMIC_RELAXED); }
    void clear(int nt) {
        int64_t nw = (int64_t)w.size();
#pragma omp parallel for num_threads(nt) schedule(static)
        for (int64_t i = 0; i < nw; ++i) w[i] = 0;
    }
};

// One bottom-up level: every unvisited vertex looks for *any* parent in the
// frontier and stops at the first hit. That early exit is the whole point:
// when the frontier is huge, most unvisited vertices find a parent after a
// few probes instead of the frontier scanning all of its edges.
// Only v's own iteration writes dist[v], so no atomics on dist are needed.
StepResult bottomUpStep(const graph::CSR& g, uint32_t* dist, const Bitmap& front, Bitmap& next,
                        uint32_t d, int nt) {
    const int64_t n = (int64_t)g.n;
    uint64_t awake = 0, examined = 0;
#pragma omp parallel for num_threads(nt) schedule(dynamic, 1024) reduction(+ : awake, examined)
    for (int64_t v = 0; v < n; ++v) {
        if (dist[v] != kUnreached) continue;
        const uint32_t *b = g.begin((uint32_t)v), *e = g.end((uint32_t)v), *p = b;
        for (; p != e; ++p) {
            if (front.get(*p)) {
                dist[v] = d;
                next.setAtomic((uint32_t)v);
                ++awake;
                ++p;
                break;
            }
        }
        examined += (uint64_t)(p - b);
    }
    return {awake, 0, examined};
}

void queueToBitmap(const uint32_t* q, int64_t size, Bitmap& bm, int nt) {
    bm.clear(nt);
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int64_t i = 0; i < size; ++i) bm.setAtomic(q[i]);
}

int64_t bitmapToQueue(const Bitmap& bm, uint64_t n, uint32_t* q, int nt) {
    uint64_t tail = 0;
    const int64_t nw = (int64_t)bm.w.size();
#pragma omp parallel num_threads(nt)
    {
        std::vector<uint32_t> local;
        local.reserve(kLocal);
        auto flush = [&] {
            if (local.empty()) return;
            uint64_t pos = __atomic_fetch_add(&tail, (uint64_t)local.size(), __ATOMIC_RELAXED);
            std::memcpy(q + pos, local.data(), local.size() * sizeof(uint32_t));
            local.clear();
        };
#pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < nw; ++i) {
            uint64_t word = bm.w[i];
            while (word) {
                uint32_t v = (uint32_t)(i * 64 + __builtin_ctzll(word));
                word &= word - 1;
                if (v < n) {
                    local.push_back(v);
                    if (local.size() == kLocal) flush();
                }
            }
        }
        flush();
    }
    return (int64_t)tail;
}

void setRuntimeSchedule(Schedule s) {
    switch (s) {
        case Schedule::Static: omp_set_schedule(omp_sched_static, 0); break;
        case Schedule::Dynamic64: omp_set_schedule(omp_sched_dynamic, 64); break;
        case Schedule::Guided: omp_set_schedule(omp_sched_guided, 0); break;
    }
}

} // namespace

// ------------------------------------------------------------------ cpu_omp

void bfsOmp(const graph::CSR& g, uint32_t source, uint32_t* dist, const BfsOptions& opt,
            std::vector<util::BfsLevel>* trace) {
    const int nt = threadCount(opt);
    setRuntimeSchedule(opt.schedule);
    parallelFill(dist, (int64_t)g.n, kUnreached, nt);
    std::vector<uint32_t> front(g.n), next(g.n);
    dist[source] = 0;
    front[0] = source;
    int64_t fsize = 1;
    for (uint32_t level = 0; fsize > 0; ++level) {
        double t0 = trace ? omp_get_wtime() : 0;
        StepResult r = topDownStep(g, dist, front.data(), fsize, next.data(), level + 1, nt);
        if (trace) trace->push_back({level, (uint64_t)fsize, r.examined, 'T', 'C', (omp_get_wtime() - t0) * 1e3});
        front.swap(next);
        fsize = (int64_t)r.next;
    }
}

// ------------------------------------------------------------------ cpu_do

void bfsDO(const graph::CSR& g, uint32_t source, uint32_t* dist, const BfsOptions& opt,
           std::vector<util::BfsLevel>* trace) {
    const int nt = threadCount(opt);
    const uint64_t n = g.n;
    omp_set_schedule(omp_sched_dynamic, 64);  // top-down steps
    parallelFill(dist, (int64_t)n, kUnreached, nt);

    std::vector<uint32_t> queue(n), nextQueue(n);
    Bitmap front(n), next(n);
    dist[source] = 0;
    queue[0] = source;
    int64_t fsize = 1;
    uint32_t level = 0;

    // m_u: edges not yet explored from the top-down side; m_f: edges of the
    // current frontier ("scout count"). Switch to bottom-up when the frontier
    // would examine more than 1/alpha of what is left.
    uint64_t edgesToCheck = g.m;
    uint64_t scout = g.degree(source);

    while (fsize > 0) {
        if ((double)scout > (double)edgesToCheck / opt.alpha) {
            double t0 = omp_get_wtime();
            queueToBitmap(queue.data(), fsize, front, nt);
            double convMs = (omp_get_wtime() - t0) * 1e3;
            uint64_t awake = (uint64_t)fsize, oldAwake;
            // Stay bottom-up while the frontier is growing or still large
            // (more than n/beta vertices).
            do {
                oldAwake = awake;
                double t1 = omp_get_wtime();
                next.clear(nt);
                StepResult r = bottomUpStep(g, dist, front, next, level + 1, nt);
                if (trace)
                    trace->push_back({level, oldAwake, r.examined, 'B', 'C',
                                      (omp_get_wtime() - t1) * 1e3 + convMs});
                convMs = 0;
                awake = r.next;
                std::swap(front.w, next.w);
                ++level;
            } while (awake >= oldAwake || (double)awake > (double)n / opt.beta);
            double t2 = omp_get_wtime();
            fsize = bitmapToQueue(front, n, queue.data(), nt);
            if (trace && !trace->empty()) trace->back().time_ms += (omp_get_wtime() - t2) * 1e3;
            scout = 1;
        } else {
            double t0 = trace ? omp_get_wtime() : 0;
            edgesToCheck -= std::min(edgesToCheck, scout);
            StepResult r = topDownStep(g, dist, queue.data(), fsize, nextQueue.data(), level + 1, nt);
            if (trace) trace->push_back({level, (uint64_t)fsize, r.examined, 'T', 'C', (omp_get_wtime() - t0) * 1e3});
            scout = r.scout;
            queue.swap(nextQueue);
            fsize = (int64_t)r.next;
            ++level;
        }
    }
}

} // namespace cpu
