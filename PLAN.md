# Project Plan: Graph Analytics on Apple Silicon (BFS + PageRank in Metal)

## 0. Working instructions (read first)

You are building a complete research-style project: high-performance BFS and PageRank on an Apple M2 GPU using Metal, with CPU baselines, careful benchmarking, and a final report with figures.

**Working rules**
1. Work **one phase at a time**, in order. At the end of each phase: build with zero warnings, run all tests, update `PROGRESS.md`, `git commit`, then **stop and give me a short summary** (what was built, test status, key numbers, any problems). Wait for my go-ahead before starting the next phase.
2. **Correctness before speed.** Every GPU variant must match the CPU reference exactly (BFS distances) or within tolerance (PageRank) before it is benchmarked. If a result mismatches, stop and debug. Never loosen a tolerance without explaining why in `PROGRESS.md`.
3. Respect the machine limits in Section 1. Check `df -h .` before any download. Never commit anything in `data/`.
4. Do not add dependencies beyond those listed in Section 2 without asking me.
5. Keep `PROGRESS.md` as a running log: date, phase, what was done, results, open issues, decisions made. A fresh session must be able to continue from `PLAN.md` + `PROGRESS.md` alone.
6. Prefer simple, readable code with comments explaining *why* each optimization exists. This is a learning project; the code is part of the deliverable.

---

## 1. Machine and constraints

| Item | Value | Implication |
|---|---|---|
| Chip | Apple M2 | Metal 3, GPU family Apple8 |
| GPU | 10 cores, 32-wide simdgroups, ≤1024 threads/threadgroup, 32 KB threadgroup memory | Verify via Metal API at startup and print |
| CPU | 8 cores: 4 performance + 4 efficiency | Matters for OpenMP scheduling experiments |
| Memory | 8 GB unified (shared by CPU and GPU) | Keep peak process memory **< 3 GB** |
| Bandwidth | ~100 GB/s theoretical | Measure the real value in Phase 3 |
| Free disk | ~20 GB | Whole project, including data and venv, must stay **< 4 GB** |
| Toolchain | Command Line Tools only (**no full Xcode**) | The offline `metal` shader compiler is NOT available. Compile shaders **at runtime** with `newLibraryWithSource:options:error:` |

Verify these at the start (Phase 0) with `sysctl`, `system_profiler SPDisplaysDataType`, and the Metal API rather than assuming.

---

## 2. Tech stack

- **Host code:** C++17, with a thin **Objective-C++ (`.mm`)** layer for Metal (uses the system Metal framework; no extra downloads).
- **Shaders:** Metal Shading Language, `.metal` files in `src/shaders/`, read from disk and compiled at runtime. Runtime compilation does not resolve local `#include`s, so the host must **concatenate `src/shaders/common.h` + the kernel file** into one source string before compiling. Disable fast math so PageRank results are reproducible.
- **Build:** plain `Makefile` with Apple clang (no CMake). Flags: `-std=c++17 -O3 -Wall -Wextra -march=native` (drop `-march` if clang rejects it). Link `-framework Metal -framework Foundation -lz`.
- **OpenMP:** `brew install libomp`; compile with `-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include`, link `-L$(brew --prefix libomp)/lib -lomp`.
- **gzip:** zlib from the macOS SDK (`-lz`). Read `.txt.gz` directly; never decompress to disk.
- **Python (analysis only):** venv in `.venv/` with `numpy pandas matplotlib networkx scipy`.
- **Version control:** git. `.gitignore`: `data/`, `build/`, `.venv/`, `*.bin`, `*.gz`.

---

## 3. Repository layout

```
graph-gpu/
  PLAN.md  PROGRESS.md  README.md  Makefile  .gitignore
  scripts/
    env_check.sh        # prints chip, cores, memory, disk, tool versions
    setup_python.sh     # creates .venv
    download.sh         # downloads + converts datasets, deletes .gz after
    run_all.sh          # full benchmark matrix -> results/*.csv
  src/
    graph/
      csr.hpp/.cpp      # CSR struct, binary save/load (mmap), validation
      convert.cpp       # gconvert tool: SNAP .txt.gz -> .bin
      transform.hpp/.cpp# transpose, symmetrize, permute/relabel
      reorder.hpp/.cpp  # random, degree-sort, RCM orderings
      generators.hpp/.cpp # synthetic test graphs
      stats.cpp         # gstats tool
    cpu/
      bfs_cpu.hpp/.cpp  # serial, OpenMP top-down, OpenMP direction-optimizing
      pagerank_cpu.hpp/.cpp
    metal/
      context.hpp/.mm   # device, runtime shader compile, pipeline cache, buffers, GPU timing
      bfs_gpu.hpp/.mm   # all GPU BFS variants + hybrid
      pagerank_gpu.hpp/.mm
      primitives.hpp/.mm# scan, reduce, fill (reused by BFS/PR)
    shaders/
      common.h  primitives.metal  bfs.metal  pagerank.metal  micro.metal
    bench/
      main.cpp          # gbench CLI
      micro.mm          # bandwidth + launch-overhead microbenchmarks
    util/
      timer.hpp  csv.hpp  rng.hpp  args.hpp
  tests/
    test_main.cpp       # gtest-free tiny test runner (assert-based)
  tools/
    validate_networkx.py  # cross-check on small graphs
  analysis/
    plots.py            # all figures from results/*.csv
  data/                 # .bin graphs (gitignored)
  results/              # CSV outputs
  figures/              # PNG/SVG plots
  docs/REPORT.md        # final write-up
```

Executables (in `build/`): `gconvert`, `gstats`, `gbench`, `gmicro`, `gtest`.

---

## 4. Data

### 4.1 Datasets (SNAP)

| Name | URL | Type | Notes |
|---|---|---|---|
| ego-Facebook | https://snap.stanford.edu/data/facebook_combined.txt.gz | undirected | Tiny; debugging |
| web-Google | https://snap.stanford.edu/data/web-Google.txt.gz | directed | Fits largely in cache |
| roadNet-CA | https://snap.stanford.edu/data/roadNet-CA.txt.gz | undirected | High diameter, low degree: the contrast case |
| soc-Pokec | https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz | directed | Medium social |
| soc-LiveJournal1 | https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz | directed | Main benchmark graph |
| com-Orkut | https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz | undirected | Largest; convert last with other apps closed |

If a URL 404s, find the current link on the SNAP dataset page and note it in `PROGRESS.md`.

### 4.2 Binary CSR format (`.bin`)

```
struct Header {
  char     magic[8];     // "GGPUCSR1"
  uint32_t version;      // 1
  uint32_t flags;        // bit0: directed, bit1: sorted adjacency, bit2: deduplicated
  uint64_t num_vertices; // n
  uint64_t num_edges;    // m (number of entries in col_idx)
};
uint32_t row_ptr[n+1];
uint32_t col_idx[m];
```
- Vertex IDs are relabeled to dense `0..n-1` (SNAP IDs can be sparse or 1-based).
- Self-loops removed, duplicate edges removed, adjacency lists sorted.
- Assert `m < 2^32`. All graphs here fit in uint32.
- Load with `mmap`. **Optional optimization:** wrap page-aligned mmap'd arrays with `newBufferWithBytesNoCopy` (Apple Silicon page size is 16 KB; length must be a page multiple). Fall back to copying if alignment fails.

### 4.3 Memory-efficient converter (important for 8 GB)

Do **not** materialize an edge list of uint64 pairs and sort it. That would need ~2 GB for Orkut. Instead:
1. **Pass 1** (stream the .gz): find max ID / build ID remap, count degrees (for undirected inputs, count both directions).
2. Prefix-sum degrees into `row_ptr`.
3. **Pass 2** (stream the .gz again): scatter neighbors into `col_idx` using per-vertex cursors.
4. Sort each adjacency list, remove duplicates and self-loops, compact `col_idx`, rebuild `row_ptr`. Parallelize with OpenMP.
5. Write `.bin`, then delete the `.gz` (in `download.sh`, after verifying the `.bin` loads).

### 4.4 Derived graphs (built in memory at load time, never stored)

- **BFS** always runs on the **symmetrized** graph (standard practice, e.g. GAP `-s`). For undirected inputs this is the graph itself.
- **PageRank** runs on the **directed** graph: needs out-degrees and the **transpose** (in-edges, for pull). For undirected graphs the transpose is the graph itself.
- Transform cost is not part of any timed region.

### 4.5 Synthetic test graphs (`generators.cpp`)

Path (n=1000), star (1 hub + 100k leaves: extreme imbalance), complete graph (n=200), two disconnected components, empty graph with isolated vertices, 2D grid (road-like), small R-MAT/Kronecker (power-law, Graph500 parameters a=0.57, b=0.19, c=0.19). These are the core unit-test inputs.

---

## 5. Algorithms and variants

Every variant is selectable by name from `gbench --variant <name>`.

### 5.1 BFS

Output: `dist[v]` (level from source, `0xFFFFFFFF` = unreached). Optionally `parent[v]` for validation.

| Variant | Description |
|---|---|
| `cpu_serial` | Queue-based BFS. The reference implementation. |
| `cpu_omp` | Level-synchronous top-down, OpenMP. Schedule configurable: `static`, `dynamic,64`, `guided`. Thread count configurable (1–8). |
| `cpu_do` | OpenMP direction-optimizing BFS (Beamer et al., SC'12): top-down with queue, bottom-up with bitmap; switch using α=15, β=18 as in the GAP Benchmark Suite. |
| `gpu_vertex` | One thread per **vertex** each level; thread works only if `dist[v]==level`. No queue. Simple baseline that wastes threads. |
| `gpu_naive` | One thread per **frontier vertex**; claim neighbors with `atomic_compare_exchange_weak_explicit` on `dist`; append to next frontier with `atomic_fetch_add`. Exposes load imbalance. |
| `gpu_binned` | Load-balanced (Merrill et al., PPoPP'12 style). Each level: classify frontier vertices by degree into 3 queues: small (<32 → thread per vertex), medium (32–1023 → **simdgroup** per vertex, lanes stride the neighbor list), large (≥1024 → **threadgroup** per vertex). Three kernel dispatches per level. |
| `gpu_edge` | Edge-balanced: exclusive scan of frontier degrees, then each thread handles a fixed number of edges, finding its source vertex by binary search into the scan (merge-path style). Uses the scan primitive. |
| `gpu_do` | GPU direction-optimizing: top-down uses the best of `gpu_binned`/`gpu_edge`; bottom-up uses a **bitmap frontier**: one thread per unvisited vertex scans its neighbors and **exits early** at the first frontier hit. Kernels to convert queue↔bitmap. Track `m_f` (edges from frontier, via reduction) and `m_u` (unexplored edges) for the α/β heuristic. |
| `hybrid` | Same as `gpu_do`, but levels whose frontier work (edges to examine) is below a threshold `T` run **on the CPU** (OpenMP), directly on the same shared buffers. Zero copying thanks to unified memory. `T` is a CLI parameter; Phase 7 sweeps it. |

**Per-level control:** the host reads the next-frontier size after each level (`waitUntilCompleted`, then read the shared buffer). This sync is a real cost and is part of what we measure. **Stretch goal:** use indirect dispatch (`dispatchThreadgroupsWithIndirectBuffer`) with a small kernel that computes the next grid size on the GPU, encoding several levels per command buffer to reduce syncs.

**Trace mode (`--trace`):** per level, record frontier size, edges examined, direction (TD/BU), device (CPU/GPU), and time. Written to CSV. This produces some of the most important figures.

### 5.2 PageRank

Definition (fixed for all variants): damping `d = 0.85`;
`r_new[v] = (1-d)/n + d * ( Σ_{u ∈ in(v)} r[u]/outdeg[u]  +  dangling_sum/n )`, where `dangling_sum` = total rank on vertices with outdegree 0. Initial `r = 1/n`. Use `float` on GPU (Metal has no double) and `double` in the CPU reference.

Per iteration on GPU: (1) `contrib[u] = r[u]/outdeg[u]` and dangling-sum reduction; (2) main SpMV-like kernel; (3) L1-difference reduction (only in convergence mode).

| Variant | Description |
|---|---|
| `cpu_serial` | Pull-based reference, double precision. |
| `cpu_omp` | Pull-based, OpenMP (`dynamic` schedule). |
| `gpu_push` | Thread per source vertex, adds `contrib` to each out-neighbor with atomic float add (`atomic_float` + `atomic_fetch_add_explicit`, supported in MSL 3.0 on Apple8). If that fails to compile, fall back to a compare-exchange loop on the bit pattern. Expect contention on hubs. |
| `gpu_pull_scalar` | Thread per vertex over in-edges (transpose CSR). No atomics. |
| `gpu_pull_vector` | **Simdgroup per vertex**, lanes stride in-edges, combine with `simd_sum`. |
| `gpu_pull_adaptive` | Bin vertices once by in-degree (bins are fixed across iterations), thread/simdgroup/threadgroup per vertex as in `gpu_binned`. |

**Modes:** `--iters 20` (fixed iterations; report ms per iteration, the benchmark metric) and `--tol 1e-6` (run to convergence; report iteration count).

### 5.3 Vertex reordering (`reorder.cpp`)

Relabel vertices with a permutation, then rebuild CSR (`transform::permute`). Orderings:
- `original` (as converted)
- `random` (worst-case locality; shows how much the original order helps)
- `degree` (hub sort: descending degree)
- `rcm` (Reverse Cuthill-McKee: BFS from a low-degree peripheral vertex, visiting neighbors in ascending degree, reverse the final order)

Also compute a locality metric for each ordering: mean `|u - v|` over edges, and the fraction of edges whose endpoints fall within the same 64-vertex block. Results map back to original IDs before validation.

---

## 6. Metal implementation notes

- Create all buffers with `MTLResourceStorageModeShared`. Unified memory means CPU and GPU see the same bytes; after `waitUntilCompleted` the CPU can read results directly.
- `context.mm` provides: device init (print name, `maxThreadsPerThreadgroup`, `maxBufferLength`, `recommendedMaxWorkingSetSize`), runtime compilation with a pipeline cache keyed by function name, buffer helpers, and a `dispatch()` helper.
- **GPU timing:** use `commandBuffer.GPUStartTime`/`GPUEndTime` for kernel-only time, and a host wall-clock timer for end-to-end time. **End-to-end is the headline metric** for BFS (it includes per-level syncs); report both.
- Device atomics support only `memory_order_relaxed`. Ensure the algorithms don't need more.
- Simdgroup intrinsics to use: `simd_sum`, `simd_prefix_exclusive_sum`, `simd_ballot`, `simd_broadcast`; `threads_per_simdgroup` = 32 (assert at runtime).
- Threadgroup size is a tunable parameter (64/128/256/512/1024). Phase 10 sweeps it for the main kernels.
- **Primitives** (`primitives.metal`): fill, reduce (sum/max), exclusive scan of uint32 (three-phase: per-threadgroup scan → scan of block sums → add back), queue↔bitmap conversion. Unit-test each against CPU on random inputs, including sizes that are not multiples of the threadgroup size, and size 0 and 1.
- Never let a kernel read past the end of a buffer: every kernel takes its element count and bounds-checks.

---

## 7. Validation (non-negotiable)

- **BFS:** GPU `dist` must equal `cpu_serial` `dist` **exactly** for every vertex and every tested source. If parents are produced, check each parent is a neighbor at level `dist-1`.
- **PageRank:** compare to `cpu_serial` (double): L1 norm of difference < 1e-4 and max relative error on the top-1000 vertices < 1e-3. Ranks must sum to 1 ± 1e-4.
- **NetworkX cross-check** (`tools/validate_networkx.py`): on ego-Facebook, web-Google and all synthetic graphs, confirm `cpu_serial` BFS matches `networkx.single_source_shortest_path_length` and PageRank matches `networkx.pagerank(alpha=0.85, tol=1e-10)` within tolerance. This validates the reference itself.
- `gbench --validate` runs the CPU reference and checks the chosen variant. `run_all.sh` validates every variant on every dataset once before timing it.
- `make test` runs all unit tests (converter round-trip, transforms, reorder is a valid permutation, primitives, every BFS/PR variant on every synthetic graph).

---

## 8. Benchmark methodology

- **BFS sources:** 64 random sources with degree > 0, fixed seed (Graph500 style). Same sources for every variant.
- **GTEPS denominator:** number of undirected edges in the connected component reached from the source (computed from the reference result). Same denominator for all variants, including direction-optimizing ones that examine fewer edges. Report harmonic mean of TEPS across sources (as in Graph500), plus median time.
- **Trials:** 2 warm-up runs, then ≥5 timed runs; report median and min.
- **Environment:** plugged in to power, Low Power Mode off, other apps closed. `run_all.sh` prints a reminder and records `uptime` load average in the CSV.
- **CSV columns** (all results): `timestamp, git_commit, chip, dataset, n, m, ordering, algo, variant, params, source, trial, time_ms_e2e, time_ms_gpu, gteps, iters, validated`.

---

## 9. Phases

Each phase lists **tasks** and **acceptance criteria**. Do not start a phase until the previous one meets its criteria and I approve.

### Phase 0: Environment and scaffold
- `scripts/env_check.sh`: chip name, CPU core counts (`sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu`), memory, free disk, clang version, libomp presence, Python version. Install libomp via brew if missing (ask first).
- `scripts/setup_python.sh`, `.gitignore`, `Makefile`, `README.md` skeleton, `PROGRESS.md`, git init.
- Metal hello world in `gmicro --hello`: runtime-compile a vector-add kernel, run on 10M floats, verify, and print device properties.
- An OpenMP hello world that prints thread count.

**Accept:** `make` builds with zero warnings; vector-add verified; device properties printed and recorded in `PROGRESS.md`.

### Phase 1: Graph I/O, converter, datasets
- `csr`, `convert` (two-pass streaming, Section 4.3), `transform` (transpose, symmetrize), `generators`, `gstats`.
- `gstats` reports: n, m, directed?, min/mean/max degree, degree histogram (log2 buckets, to CSV), number of connected components (symmetrized), approximate diameter (max BFS depth from 10 random sources). Peak RSS of converter.
- `scripts/download.sh`: for each dataset: check disk, download, convert, verify load, delete `.gz`. Convert com-Orkut last.
- Unit tests: round-trip save/load; transpose of transpose = original; symmetrize is symmetric.

**Accept:** all six `.bin` files exist; `data/` < 2 GB; `results/dataset_stats.csv` written; converter peak memory < 3 GB on Orkut (measure with `/usr/bin/time -l`).

### Phase 2: CPU BFS baselines
- `cpu_serial`, `cpu_omp`, `cpu_do`; `gbench` CLI skeleton with CSV output; NetworkX validation script.
- **P-core/E-core experiment:** `cpu_omp` with schedules `static`, `dynamic,64`, `guided` at 1, 2, 4, 6, 8 threads on LiveJournal and roadNet-CA.

**Accept:** all CPU variants validated against `cpu_serial` and NetworkX; scaling CSV produced.

### Phase 3: Microbenchmarks
- `gmicro --bandwidth`: GPU read, write, and copy kernels over buffer sizes 1 MB → 2 GB (powers of 2); CPU equivalents with OpenMP. Report GB/s. This should reveal the system-level cache effect at small sizes and the true DRAM bandwidth at large sizes.
- `gmicro --launch`: time 1000 empty-kernel dispatches with `waitUntilCompleted` each (per-dispatch round-trip latency), and 1000 dispatches in one command buffer.
- `gmicro --random`: GPU random-gather bandwidth (read `a[idx[i]]` with random `idx`) vs sequential, over sizes. This is the access pattern graphs actually have.

**Accept:** `results/micro_*.csv`; measured peak sequential bandwidth and launch latency recorded in `PROGRESS.md`. These numbers are used as reference points for all later analysis.

### Phase 4: Metal infrastructure + first GPU BFS
- `context.mm` (pipeline cache, timing), `primitives` (fill, reduce, scan, conversions) with unit tests.
- `gpu_vertex` and `gpu_naive`, with `--trace`.

**Accept:** both variants pass validation on all synthetic and real graphs; first GTEPS numbers in `PROGRESS.md`.

### Phase 5: Load-balanced GPU BFS
- `gpu_binned` and `gpu_edge`. Tune the small/medium/large thresholds and threadgroup sizes.
- Star graph is the key test: `gpu_naive` should be dramatically slower on it than the balanced variants.

**Accept:** validated; comparison table of `gpu_naive` vs `gpu_binned` vs `gpu_edge` across datasets.

### Phase 6: Direction-optimizing GPU BFS
- `gpu_do` with bitmap bottom-up and α/β switching; trace shows direction per level.
- Sweep α ∈ {5, 10, 15, 20, 30}, β ∈ {10, 18, 24} on LiveJournal and Pokec; pick defaults.

**Accept:** validated; speedup over best top-down GPU variant on social graphs reported; traces for LiveJournal and roadNet-CA saved.

### Phase 7: Hybrid CPU+GPU BFS (the unified-memory experiment)
- `hybrid` variant: levels with frontier work below `T` run on CPU with OpenMP over the same shared buffers.
- Sweep `T` from 0 (all GPU) to ∞ (all CPU) on log scale, on roadNet-CA, LiveJournal and web-Google.
- Use the Phase 3 launch latency to predict a sensible `T`, then compare the prediction with the sweep result.
- Stretch: indirect-dispatch multi-level GPU BFS; compare with hybrid.

**Accept:** validated; threshold-sweep CSV; per-level trace showing which device ran each level; a short written explanation in `PROGRESS.md` of why the best `T` is what it is.

### Phase 8: PageRank
- CPU reference and OpenMP; all four GPU variants; fixed-iteration and convergence modes.
- Estimate **effective bandwidth** for pull variants: bytes = `4(n+1)` (row_ptr) + `4m` (col_idx) + `4m` (contrib gathers) + `4n` (writes), divided by time per iteration. Report as % of Phase 3 measured sequential and random-gather bandwidth.

**Accept:** validated on all datasets; per-iteration time table; push vs pull and scalar vs vector conclusions recorded.

### Phase 9: Vertex reordering
- Implement orderings; for each dataset × ordering, run `gpu_pull_vector`/`gpu_pull_adaptive` PageRank and the best GPU BFS. Record locality metrics alongside.

**Accept:** results validated (after mapping back to original IDs); table of speedup vs `original` and vs `random`.

### Phase 10: Full benchmark run, figures, report
- Threadgroup-size sweep for the main kernels; lock in best defaults.
- `scripts/run_all.sh`: full matrix (all datasets × all variants × orderings where relevant), validate-then-time. Print an estimated runtime first.
- `analysis/plots.py` produces (PNG + SVG in `figures/`):
  1. Bandwidth vs buffer size (CPU vs GPU; sequential vs random gather)
  2. Degree distributions of all datasets (log-log)
  3. Frontier size per BFS level: LiveJournal vs roadNet-CA (log y)
  4. BFS GTEPS: all variants × datasets (grouped bars)
  5. Per-level time breakdown for `gpu_do` vs `hybrid` on roadNet-CA
  6. Hybrid threshold sweep
  7. OpenMP scaling with static/dynamic/guided (P/E-core effect)
  8. PageRank ms/iteration: variants × datasets
  9. Reordering speedups, with locality metric annotated
  10. GTEPS vs graph size (cache effect)
  11. Achieved bandwidth as % of measured peak for the best BFS and PageRank variants
- `docs/REPORT.md`: abstract; hardware; datasets table; method; one section per experiment (figure + 1–3 paragraph interpretation explaining the *why*); efficiency comparison stated as GTEPS per GB/s of bandwidth (don't compare raw GTEPS to data-center GPU papers); limitations; future work; references (Beamer SC'12, Merrill PPoPP'12, GAP Benchmark Suite, Graph500, SNAP).
- Final `README.md`: what the project is, headline results, how to build, download data, run tests, reproduce every figure.

**Accept:** `make test` passes; `run_all.sh` completes; all figures generated; report complete; repository clean and committed.

### Phase 11 (optional): External reference
- Clone the GAP Benchmark Suite (github.com/sbeamer/gapbs), adapt its Makefile for Apple clang + libomp, run its `bfs` and `pr` on the same graphs, and add it to the comparison as a "state-of-the-art CPU" baseline.

---

## 10. Expected results (rough hypotheses to test, not targets)

- Naive GPU BFS will be well under 1 GTEPS on LiveJournal and suffer on the star graph; load balancing and direction-optimization should give large multiples on social graphs.
- On roadNet-CA, GPU-only BFS will be dominated by per-level sync overhead (hundreds of levels); hybrid should win clearly.
- `static` OpenMP scheduling should underperform on this 4P+4E CPU.
- Pull PageRank should beat push; vector should beat scalar on high-degree graphs, and scalar may win on roadNet-CA.
- `random` ordering should be markedly slower than `original`; RCM/degree ordering may improve on `original`.
- Small graphs (web-Google) should show higher GTEPS due to the system-level cache.

Report what actually happens, especially where a hypothesis is wrong; that is a result too.
