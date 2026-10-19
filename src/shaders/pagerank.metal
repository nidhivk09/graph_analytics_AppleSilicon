// pagerank.metal: GPU PageRank (PLAN.md 5.2). Host side: src/metal/pagerank_gpu.{hpp,mm}.
//
// One iteration = (1) pr_contrib: contrib[u] = r[u]/outdeg[u] and the
// dangling-rank partial sums, then pr_reduce -> scalars[SC_DANGLING];
// (2) the main kernel (push or one of three pull variants) writes r_new;
// (3) tol mode only: pr_diff + pr_reduce -> scalars[SC_DELTA] (L1 change).
// float throughout (Metal has no double); the CPU reference is double.
// Two-pass reductions with fixed shapes: deterministic (except gpu_push,
// whose float atomics add in arbitrary order).

constant uint SC_DANGLING = 0;
constant uint SC_DELTA = 1;

struct PrParams {
    uint n;
    float damping;
    float base;        // (1 - d) / n
    float inv_n;       // 1 / n
    uint size;         // adaptive: vertices in this bin
};

// Compensated (Kahan) summation for the per-thread loops over in-edges.
// A plain float loop over a hub's in-list rounds once per term; on star100k
// (hub in-degree 100k, all terms equal) the errors all go the same way and
// gpu_pull_scalar ended with L1 error 1e-3 and ranks summing to 1.00047,
// outside PLAN.md 7 tolerances. The compensation term carries the lost low
// bits. Costs 3 extra flops per edge in a memory-bound loop; it is only
// correct because the shaders are compiled with fast math off.
struct Kahan {
    float sum = 0.0f, comp = 0.0f;
    void add(float x) {
        float y = x - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
};

// r_new[v] from the pulled sum: (1-d)/n + d * (sum + dangling/n).
static inline float pr_value(float s, constant PrParams& p, device const float* scalars)
{
    return p.base + p.damping * (s + scalars[SC_DANGLING] * p.inv_n);
}

kernel void pr_init(device float* r [[buffer(0)]],
                    constant PrParams& p [[buffer(1)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) r[gid] = p.inv_n;
}

// contrib[u] = r[u] / outdeg[u] (0 for dangling vertices); per-threadgroup
// partial sums of the dangling rank into partial[tgid]. Grid-stride with a
// capped grid so pass 2 has at most 1024 partials.
kernel void pr_contrib(device const float* r       [[buffer(0)]],
                       device const uint* outdeg   [[buffer(1)]],
                       device float* contrib       [[buffer(2)]],
                       device float* partial       [[buffer(3)]],
                       constant PrParams& p        [[buffer(4)]],
                       uint gid  [[thread_position_in_grid]],
                       uint grid [[threads_per_grid]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint tid  [[thread_index_in_threadgroup]],
                       uint sid  [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup float scratch[32];
    float dangling = 0;
    for (uint u = gid; u < p.n; u += grid) {
        uint d = outdeg[u];
        float ru = r[u];
        contrib[u] = d ? ru / float(d) : 0.0f;
        if (d == 0) dangling += ru;
    }
    dangling = tg_sum_f(dangling, scratch, sid, lane, nsg);
    if (tid == 0) partial[tgid] = dangling;
}

// L1 change partials: sum |a - b|.
kernel void pr_diff(device const float* a   [[buffer(0)]],
                    device const float* b   [[buffer(1)]],
                    device float* partial   [[buffer(2)]],
                    constant PrParams& p    [[buffer(3)]],
                    uint gid  [[thread_position_in_grid]],
                    uint grid [[threads_per_grid]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint tid  [[thread_index_in_threadgroup]],
                    uint sid  [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup float scratch[32];
    float s = 0;
    for (uint v = gid; v < p.n; v += grid) s += fabs(a[v] - b[v]);
    s = tg_sum_f(s, scratch, sid, lane, nsg);
    if (tid == 0) partial[tgid] = s;
}

// Pass 2: one threadgroup sums `count` partials into scalars[slot].
kernel void pr_reduce(device const float* partial [[buffer(0)]],
                      device float* scalars       [[buffer(1)]],
                      constant uint& count        [[buffer(2)]],
                      constant uint& slot         [[buffer(3)]],
                      uint tid  [[thread_index_in_threadgroup]],
                      uint tgsz [[threads_per_threadgroup]],
                      uint sid  [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup float scratch[32];
    float s = 0;
    for (uint i = tid; i < count; i += tgsz) s += partial[i];
    s = tg_sum_f(s, scratch, sid, lane, nsg);
    if (tid == 0) scalars[slot] = s;
}

// ---------------------------------------------------------------------------
// gpu_push: thread per source vertex, scatter contrib to out-neighbors with
// float atomics (MSL 3.0 atomic_float on Apple8). Every edge is an atomic
// read-modify-write on a random address; hubs' in-neighbors all hit the same
// address (contention).
//
// Simdgroup aggregation: the loop runs simdgroup-uniform rounds (max degree
// in the simdgroup) and, in rounds where every active lane targets the same
// vertex, the lanes simd_sum their values and issue ONE atomic. Besides
// cutting contention, this fixes an accuracy problem: float atomics add
// sequentially, and on star100k the hub received 100k adds of ~1e-5 into an
// accumulator near 0.46 (ulp 3e-8), all rounding the same way, giving L1
// error 1e-3 and ranks summing to 1.00047 (outside PLAN.md 7 tolerances).
// Real graphs passed (L1 ~1e-7: their rounding errors are random).
//
// Compensated atomics: aggregation alone still left the star at L1 ~1e-4,
// right at the PLAN.md 7 tolerance (the hub still takes ~3k atomic adds per
// iteration into a value near 0.46; 1 test run in 15 failed, depending on
// atomic order). Kahan cannot wrap an atomic, but each add's rounding error
// can be computed exactly afterwards: atomic_fetch_add returns the old value
// and stored fl(old + x), so TwoSum(old, x) gives the error, which goes into
// a second accumulator acc_lo (tiny magnitudes, so its own rounding is
// negligible). Result = acc + acc_lo. Requires fast math off (TwoSum).
// Compensating every add doubled the atomics and made push 2.2x slower
// (LiveJournal 33 -> 73 ms/iteration). So only adds that lose more than
// 2^-16 of the term are compensated: that needs an accumulator >~250x the
// term, i.e. hub-like vertices. Skipped errors are each < |x| * 2^-16, so
// even if all were systematic they total < 1.5e-5 of the accumulator.
static inline void add_compensated(device atomic_float* acc, device atomic_float* acc_lo, uint t, float x)
{
    float old = atomic_fetch_add_explicit(&acc[t], x, memory_order_relaxed);
    float s = old + x;          // the value the atomic stored
    float bp = s - old;         // TwoSum (Knuth): exact error of old + x
    float err = (old - (s - bp)) + (x - bp);
    if (fabs(err) > fabs(x) * 0x1p-16f) atomic_fetch_add_explicit(&acc_lo[t], err, memory_order_relaxed);
}

kernel void pr_push(device const uint* row_ptr [[buffer(0)]],
                    device const uint* col_idx [[buffer(1)]],
                    device const float* contrib [[buffer(2)]],
                    device atomic_float* acc   [[buffer(3)]],
                    constant PrParams& p       [[buffer(4)]],
                    device atomic_float* acc_lo [[buffer(5)]],
                    uint gid [[thread_position_in_grid]])
{
    bool active = gid < p.n;  // no early return: rounds are simdgroup-uniform
    float c = active ? contrib[gid] : 0.0f;
    uint begin = active ? row_ptr[gid] : 0u;
    uint deg = (active && c != 0.0f) ? row_ptr[gid + 1] - begin : 0u;  // dangling: nothing to push
    uint rounds = simd_max(deg);
    for (uint k = 0; k < rounds; ++k) {
        bool has = k < deg;
        uint t = has ? col_idx[begin + k] : 0xFFFFFFFFu;
        uint lead = simd_min(t);  // some active lane's target (inactive lanes hold UINT_MAX)
        if (simd_all(!has || t == lead)) {
            float s = simd_sum(has ? c : 0.0f);
            if (simd_is_first()) add_compensated(acc, acc_lo, lead, s);
        } else if (has) {
            add_compensated(acc, acc_lo, t, c);
        }
    }
}

// r_new = (1-d)/n + d * (acc + acc_lo + dangling/n); both accumulators are
// zeroed for the next iteration.
kernel void pr_push_finalize(device float* acc            [[buffer(0)]],
                             device float* r_new          [[buffer(1)]],
                             device const float* scalars  [[buffer(2)]],
                             constant PrParams& p         [[buffer(3)]],
                             device float* acc_lo         [[buffer(4)]],
                             uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    r_new[gid] = pr_value(acc[gid] + acc_lo[gid], p, scalars);
    acc[gid] = 0.0f;
    acc_lo[gid] = 0.0f;
}

// ---------------------------------------------------------------------------
// Pull variants over the transpose (in-neighbor lists). No atomics.

#define PULL_ARGS                                                   \
    device const uint* row_ptr   [[buffer(0)]],                     \
    device const uint* col_idx   [[buffer(1)]],                     \
    device const float* contrib  [[buffer(2)]],                     \
    device float* r_new          [[buffer(3)]],                     \
    device const float* scalars  [[buffer(4)]],                     \
    constant PrParams& p         [[buffer(5)]]

// gpu_pull_scalar: thread per vertex walks its in-list. Neighboring threads
// read unrelated lists (uncoalesced col_idx) and hubs stall their simdgroup.
kernel void pr_pull_scalar(PULL_ARGS, uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    Kahan s;
    for (uint e = row_ptr[gid]; e < row_ptr[gid + 1]; ++e) s.add(contrib[col_idx[e]]);
    r_new[gid] = pr_value(s.sum, p, scalars);
}

// gpu_pull_vector: simdgroup per vertex; lanes stride the in-list (32
// consecutive col_idx entries per round: coalesced) and simd_sum combines.
// Wastes lanes on low-degree vertices (most vertices of a power-law graph).
kernel void pr_pull_vector(PULL_ARGS,
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]])
{
    uint v = gid / 32;
    if (v >= p.n) return;  // uniform for the whole simdgroup
    Kahan k;
    for (uint e = row_ptr[v] + lane; e < row_ptr[v + 1]; e += 32) k.add(contrib[col_idx[e]]);
    float s = simd_sum(k.sum);
    if (lane == 0) r_new[v] = pr_value(s, p, scalars);
}

// gpu_pull_adaptive: vertices pre-binned by in-degree (once; bins are fixed
// across iterations). `verts` = this bin's vertex list.

// Small bin: thread per vertex (as gpu_pull_scalar, over the bin list).
// A simdgroup-cooperative version (lanes share the concatenated short lists,
// as in BFS Phase 5) was measured first and dropped: on roadNet-CA, where
// every vertex is small, it took 1.43 ms/iteration vs 0.95 for this loop,
// and it was no faster on LiveJournal or Orkut. Short PageRank lists gain
// nothing from expansion; the shuffle/prefix-sum work per round cost more.
kernel void pr_pull_small(PULL_ARGS,
                          device const uint* verts [[buffer(6)]],
                          uint gid [[thread_position_in_grid]])
{
    if (gid >= p.size) return;
    uint v = verts[gid];
    Kahan s;
    for (uint e = row_ptr[v]; e < row_ptr[v + 1]; ++e) s.add(contrib[col_idx[e]]);
    r_new[v] = pr_value(s.sum, p, scalars);
}

// Medium bin: simdgroup per vertex (as gpu_pull_vector, over the bin list).
kernel void pr_pull_medium(PULL_ARGS,
                           device const uint* verts [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]])
{
    uint i = gid / 32;
    if (i >= p.size) return;
    uint v = verts[i];
    Kahan k;
    for (uint e = row_ptr[v] + lane; e < row_ptr[v + 1]; e += 32) k.add(contrib[col_idx[e]]);
    float s = simd_sum(k.sum);
    if (lane == 0) r_new[v] = pr_value(s, p, scalars);
}

// Large bin: threadgroup per vertex, threadgroup reduction.
kernel void pr_pull_large(PULL_ARGS,
                          device const uint* verts [[buffer(6)]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint tid  [[thread_index_in_threadgroup]],
                          uint tgsz [[threads_per_threadgroup]],
                          uint sid  [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup float scratch[32];
    if (tgid >= p.size) return;  // uniform per threadgroup
    uint v = verts[tgid];
    Kahan k;
    for (uint e = row_ptr[v] + tid; e < row_ptr[v + 1]; e += tgsz) k.add(contrib[col_idx[e]]);
    float s = tg_sum_f(k.sum, scratch, sid, lane, nsg);
    if (tid == 0) r_new[v] = pr_value(s, p, scalars);
}
