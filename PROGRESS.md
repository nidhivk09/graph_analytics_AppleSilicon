# PROGRESS

Running log. A fresh session should be able to continue from `PLAN.md` + this file.

## Current state

- **Phase 13 (extension) complete** (2026-10-20): straggler fixes for `gpu_pull_gs` (pair deferral, per-vertex stopping test). Phase 12 (2026-10-19): in-place GPU PageRank. All PLAN phases 0–11 are done.
- Presentation website: `docs/index.html` (standalone, figures in `docs/figures`); also published as a private claude.ai artifact. GAP baseline: `scripts/gap.sh` (GAP in gitignored `external/gapbs`, pinned to 941462c).
- Final results: `docs/REPORT.md` (11 figures in `figures/`, all from one `scripts/run_all.sh` run on AC power).
- **Reference numbers** (Phase 3, AC power): GPU sequential DRAM bandwidth **~95 GB/s**;
  CPU (8 threads) ~61 GB/s read; dispatch round trip (commit + wait) **~111 µs**;
  dependent dispatch in one encoder **~0.6 µs** (GPU) / 0.8 µs (wall); random 4 B gather on DRAM-sized
  tables **~0.28 Greads/s GPU, ~0.32 Greads/s CPU** vs ~12 Greads/s sequential.
- All six datasets are in `data/*.bin` (1.3 GB). Rebuild them with `scripts/download.sh`.

## Measured machine facts (Phase 0)

From `scripts/env_check.sh` and `build/gmicro --hello` (Metal API):

| Item | Value |
|---|---|
| Chip | Apple M2 |
| CPU | 8 cores = 4 performance + 4 efficiency |
| Memory | 8 GB unified |
| Page size | 16384 bytes |
| GPU | Apple M2, 10 cores (system_profiler; core count is not exposed by the Metal API) |
| Metal families | Apple8: yes, Metal3: yes; unified memory: yes |
| maxThreadsPerThreadgroup | 1024 (pipeline maxTotalThreadsPerThreadgroup = 1024 for vector_add) |
| maxThreadgroupMemoryLength | 32768 bytes |
| threadExecutionWidth (simdgroup) | 32 (asserted for every pipeline in `MetalContext::pipeline`) |
| maxBufferLength | 4096 MiB |
| recommendedMaxWorkingSetSize | 5461 MiB |
| macOS / SDK | 26.2 / Command Line Tools SDK 13.3 |
| Compiler | Apple clang 14.0.3; `-march=native` accepted |
| Offline `metal` compiler | not available, as expected; shaders compiled at runtime |
| libomp | Homebrew 23.1.2 (already installed) |
| Free disk at start | 17 GiB |

vector_add on 10M floats: verified (0 mismatches); first-run GPU time 1.49 ms
(80.5 GB/s effective for 120 MB of traffic); end-to-end 18.7 ms (includes first
command-buffer setup). OpenMP: 8 threads by default.

## Log

### 2026-10-14 · Phase 0: environment and scaffold

**Done**
- `PLAN.md` (the plan verbatim), `.gitignore`, `Makefile`, `README.md` skeleton, this file, `git init`.
- `scripts/env_check.sh`: chip, P/E cores, memory, page size, GPU cores, Low Power Mode, power source, disk, toolchain, libomp, Python, venv.
- `scripts/setup_python.sh`: `.venv` with numpy 2.5.4, pandas 3.0.6, matplotlib 3.11.2, networkx 3.7, scipy 1.18.1 (306 MB).
- `src/metal/context.{hpp,mm}`: device and queue, device-info print, runtime compile of `common.h` + `<file>.metal` (fast math off, MSL 3.0), per-file library cache, pipeline cache keyed by function name, assertion that simdgroup width is 32, shared-storage buffer helpers (zero-size requests become 4-byte buffers so bindings stay valid), `dispatch1D`, `gpuTimeMs`.
- `src/shaders/common.h`, `src/shaders/micro.metal` (`vector_add`).
- `build/gmicro --hello` (Metal hello world plus device properties) and `--omp` (OpenMP hello world).
- `tests/test_main.cpp`: assert-based runner. Tests: OpenMP team size, and vector_add at n = 0, 1, 255, 256, 257, 100003.

**Results**
- `make` clean rebuild: 0 warnings. `make test`: 2/2 pass. vector_add verified.

**Decisions**
- **`-mmacosx-version-min=26.0`** in compile and link flags. Homebrew libomp is built for macOS 26, while the CLT SDK is 13.3. Without this flag, every link prints `ld: warning: dylib ... was built for newer macOS version (26.0) than being linked (13.3)`, which would violate the zero-warnings rule.
- **Python 3.13 for `.venv`, not the default `python3` (3.14).** Homebrew python@3.14 here has a broken `pyexpat`: `Symbol not found: _XML_SetAllocTrackerActivationThreshold` in `/usr/lib/libexpat.1.dylib`. That breaks `plistlib` → `platform.mac_ver()` → pip's truststore, so `venv`/`ensurepip` fails. `setup_python.sh` now picks the first interpreter that passes a pyexpat check, and `$PYTHON` overrides it. (Upstream fix: `brew reinstall python@3.14 expat`. Not done, because it changes the user's system.)
- Tests are compiled as Objective-C++ so they can drive Metal directly.
- Shader directory is baked in at build time (`GGPU_SHADER_DIR`, absolute path) and can be overridden by an environment variable of the same name.
- The project lives in its own git repo (`git init` here), nested inside the home-directory repo.

**Open issues / notes for benchmarking**
- At Phase 0, **Low Power Mode was ON and the machine was on battery**. It must be plugged in with LPM off before Phase 2 timings; `env_check.sh` reports both.

### 2026-10-14 · Phase 1: graph I/O, converter, datasets

**Done**
- `src/graph/csr.{hpp,cpp}`: CSR is a cheap-to-copy view whose arrays are kept alive by a `shared_ptr`, either to an mmap or to owned vectors. Also: `save`, `load` (mmap with magic/version/size checks, `MADV_WILLNEED`), `validate`, `equal`, `compactAdjacency` (OpenMP sort+dedup+drop self-loops, then serial in-place compaction), and `fromEdgeList` for tests and generators.
- `src/graph/snap.{hpp,cpp}`: the two-pass streaming converter (PLAN 4.3). It reads 4 MB chunks with `gzread` and parses numbers by hand. Per-raw-ID degree/seen arrays grow on demand. Vertices are relabeled densely in increasing raw-ID order, so the input's order *is* the Phase 9 "original" ordering. `src/graph/convert.cpp` is the `gconvert` CLI.
- `src/graph/transform.{hpp,cpp}`: `transpose` (serial scatter, so in-lists come out sorted) and `symmetrize` (per-vertex merge of out- and in-lists, with OpenMP count and fill passes). An undirected input is returned as-is with shared storage.
- `src/graph/generators.{hpp,cpp}`: path(1000), star(100k), complete(200), two components (path 300 + 10x10 grid), empty(1000), grid 100x100, R-MAT scale 12 / edgefactor 16 (Graph500 a/b/c) in undirected and directed form. `gen::suite()` returns them all.
- `src/cpu/bfs_cpu.{hpp,cpp}`: `bfsSerial` (the `cpu_serial` reference), written now because `gstats` needs it for the diameter estimate.
- `src/graph/stats.cpp` (`gstats`): full validation of the stored and symmetrized graph, n/m, degree min/mean/max, max in-degree, isolated vertices, connected components and the largest one, approximate diameter (max depth over 10 BFS runs, seed 1). Writes `results/dataset_stats.csv` and `results/degree_hist.csv`, with log2 buckets and kinds out/in/sym (directed) or deg (undirected).
- `scripts/download.sh`: for each dataset it checks free disk (≥ 3 GB), downloads, converts under `/usr/bin/time -l`, verifies with `gstats`, then deletes the `.gz`. Valid existing `.bin` files are skipped, and com-Orkut goes last.
- `src/util/{csv,rng}.hpp`: CSV upsert keyed by the first column, so reruns replace rows instead of duplicating them; SplitMix64 RNG.
- Tests (11 total, all pass): generator shapes and validity; save/load round-trip on all synthetic graphs; bad file rejected; transpose∘transpose = id, with an edge-level check; symmetrize gives a symmetric result containing every edge, and an undirected input shares storage; converter on a hand-written file with comments, blank lines, tabs/CRLF, sparse IDs, duplicates, a self-loop, a target-only ID and no trailing newline, in gz and plain text, directed and undirected; malformed input rejected; converter on a 1-based R-MAT text dump matches the in-memory build; known BFS answers.

**Datasets** (every URL in PLAN 4.1 worked as listed; counts match SNAP's published node/edge counts and largest-WCC sizes)

| dataset | type | n | stored m | undirected edges | max deg (out / in) | CCs (largest %) | approx diam | .bin MB | convert s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|
| ego-Facebook | undir | 4,039 | 176,468 | 88,234 | 1,045 | 1 (100%) | 7 | 0.7 | 0.02 | 12 |
| web-Google | dir | 875,713 | 5,105,039 | 4,322,051 | 456 / 6,326 | 2,746 (97.7%) | 16 | 22.8 | 0.46 | 55 |
| roadNet-CA | undir | 1,965,206 | 5,533,214 | 2,766,607 | 12 | 2,638 (99.6%) | 773 | 28.6 | 0.45 | 107 |
| soc-Pokec | dir | 1,632,803 | 30,622,564 | 22,301,964 | 8,763 / 13,733 | 1 (100%) | 10 | 123.0 | 2.13 | 168 |
| soc-LiveJournal1 | dir | 4,847,571 | 68,475,391 | 42,851,237 | 20,292 / 13,905 | 1,876 (99.9%) | 14 | 279.7 | 5.14 | 372 |
| com-Orkut | undir | 3,072,441 | 234,370,166 | 117,185,083 | 33,313 | 1 (100%) | 8 | 905.8 | 11.58 | **940** |

- Acceptance: all six `.bin` present; `data/` = 1.3 GB (< 2 GB); `results/dataset_stats.csv` written; Orkut converter peak 940 MB (< 3 GB), the same from `getrusage` and `/usr/bin/time -l`. Whole project 1.6 GB including the venv; 15 GB disk free afterwards.
- The "approx diameter" values are lower bounds from 10 sources. For example SNAP lists 8 for ego-Facebook and 849 for roadNet-CA, where we report 7 and 773.
- soc-LiveJournal1: 518,382 raw lines were duplicates or self-loops. 962 vertices are isolated after self-loop removal; they stay as vertices, and BFS source selection must require degree > 0 (the plan already says so).
- com-Orkut: raw IDs go up to 3,072,626 but only 3,072,441 appear, which confirms the dense relabeling.
- Early data point: serial BFS (inside gstats, cold pages) takes ~430 ms per source on LiveJournal and ~580 ms on Orkut. Proper timing comes in Phase 2.

**Decisions / deviations from PLAN layout**
- The converter core is in `src/graph/snap.{hpp,cpp}` (a library), with `convert.cpp` as a thin CLI, so unit tests can call the converter directly.
- Converter peak RSS is recorded by `gconvert` in `results/convert_stats.csv`, plus raw `/usr/bin/time -l` logs in `results/convert_time_<name>.txt`, rather than by `gstats`, which never runs the converter.
- `compactAdjacency` does not shrink `col_idx`; the converter writes only `row_ptr[n]` entries. That avoids a second ~940 MB allocation on Orkut.
- **`newBufferWithBytesNoCopy` not used yet.** The 32-byte header means `row_ptr` and `col_idx` in the mmap are not 16 KB page-aligned, and the format is fixed by the plan. For Phase 4 the options are (a) copy into shared MTLBuffers (Orkut: ~0.9 GB mmap + 0.9 GB buffer, under 3 GB but wasteful), or (b) `pread` the file straight into page-aligned MTLBuffer contents and skip the mmap for GPU runs. **(b) is preferred.** I'll decide in Phase 4.

**Open issues**
- Low Power Mode / battery check still needed before Phase 2 timings.

### 2026-10-14 · Phase 2: CPU BFS baselines

**Done**
- `src/cpu/bfs_cpu.{hpp,cpp}`: `bfsSerial` (reference), `bfsOmp` and `bfsDO`. All variants write into a caller-provided `dist`, so the output allocation is untimed while initialization is timed.
  - `cpu_omp`: level-synchronous top-down. Neighbors are claimed with a relaxed CAS after a plain-load pre-check, and the next frontier is filled through thread-local 4096-entry buffers with one `fetch_add` per flush. The schedule is set via `omp_set_schedule` + `schedule(runtime)`.
  - `cpu_do`: Beamer direction-optimizing BFS with the GAP control flow. Go TD→BU when `scout > edges_to_check/alpha`; stay BU while `awake >= old_awake || awake > n/beta`. Bottom-up uses a bitmap frontier, an early exit at the first parent found, and no atomics on `dist`. Queue↔bitmap conversions are counted in the level time.
- `src/util/trace.hpp` (`BfsLevel`: level, frontier, edges examined, direction, device, ms), shared with the future GPU and hybrid code. `cpu_omp` and `cpu_do` fill it.
- `src/bench/main.cpp` (`gbench`): variant registry; `--graph file.bin | synth:NAME`; fixed-seed sources; warm-up and trials; `--validate` (exact check against cpu_serial on every trial); CSV in the PLAN 8 schema; `--trace`; `--dump-dist`; `--save-graph`. Summary line: median and min time, plus the harmonic mean of per-source GTEPS (from per-source median time).
- `src/util/sysinfo.hpp`: timestamp, chip, git commit (with `+` suffix if the tree is dirty), load average.
- `tools/validate_networkx.py`: reads `.bin` with numpy and compares `gbench --dump-dist` of cpu_serial with `networkx.single_source_shortest_path_length`. Writes `results/validate_networkx.csv`. (The PageRank half comes in Phase 8.)
- `scripts/scaling_cpu.sh` → `results/scaling_cpu_bfs.csv`.
- Tests (13 total, all pass): cpu_omp × {static, dynamic/64, guided} × {1, 3, 8} threads, and cpu_do in {default, all top-down, all bottom-up, frequent switching} × {1, 8} threads, on every synthetic graph from 6 sources each, including isolated vertices. Distances must be exact, and trace frontier sizes must sum to the number of reached vertices.

**Validation**
- All CPU variants match cpu_serial exactly on all 6 datasets × 64 sources, and on every row of every CSV produced in this phase.
- NetworkX: **35/35 checks OK**, with 0 mismatches. That's 3–4 sources on each of the 8 synthetic graphs, ego-Facebook and web-Google (NetworkX peak RSS 1.24 GB on web-Google).

**CPU baseline** (`results/bfs_cpu.csv`: 64 sources × (2 warm-up + 5 trials), 8 threads, cpu_omp dynamic/64, cpu_do α=15 β=18). AC power, Low Power Mode off.

| dataset | serial ms | omp ms | do ms | serial GTEPS | omp GTEPS | do GTEPS |
|---|---|---|---|---|---|---|
| ego-Facebook | 0.10 | 0.32 | 0.38 | 0.868 | 0.272 | 0.223 |
| web-Google | 32.6 | 15.2 | 6.9 | 0.129 | 0.280 | 0.592 |
| roadNet-CA | 37.5 | 39.3 | 41.3 | 0.073 | 0.069 | 0.067 |
| soc-Pokec | 162.3 | 54.7 | 8.8 | 0.137 | 0.403 | 2.483 |
| soc-LiveJournal1 | 424.7 | 138.7 | 22.4 | 0.101 | 0.301 | 1.864 |
| com-Orkut | 554.8 | 221.9 | 28.0 | 0.213 | 0.525 | **3.987** |

**P-core/E-core scaling** (`results/scaling_cpu_bfs.csv`: 16 sources × 3 trials, median ms)

| roadNet-CA | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| static | 46.4 | 27.2 | 21.0 | 34.8 | 38.3 |
| dynamic/64 | 46.9 | 28.2 | 20.8 | 34.1 | 38.0 |
| guided | 46.7 | 27.1 | 20.8 | 34.3 | 39.7 |
| cpu_do | 46.7 | 28.5 | 20.8 | 33.6 | 39.3 |

| soc-LiveJournal1 | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| static | 563.0 | 357.4 | 226.3 | 180.4 | 166.9 |
| dynamic/64 | 561.3 | 298.8 | 184.1 | 160.0 | 140.8 |
| guided | 568.6 | 298.6 | 185.5 | 162.4 | 136.1 |
| cpu_do | 91.3 | 48.8 | 30.3 | 26.1 | 22.5 |

Serial: roadNet-CA 37.9 ms, LiveJournal 424.7 ms.

**Interpretation**
- *Static scheduling underperforms* on LiveJournal by 1.13–1.23× against the best of dynamic and guided. That confirms the PLAN hypothesis, but the cause is mostly **degree imbalance**, not just P/E asymmetry: the penalty is already 1.20–1.23× at 2 and 4 threads, where all threads fit on P-cores. Static gives each thread a contiguous block of frontier vertices, and on a power-law graph those blocks have very unequal edge counts. On roadNet-CA (max degree 12) the schedule makes no difference.
- *E-cores hurt high-diameter graphs.* roadNet-CA is fastest at 4 threads (20.8 ms) and gets slower at 6 (34) and 8 (38) threads with every schedule. Its ~600–800 levels each process tiny frontiers, so per-level fork/join barrier cost dominates. Once threads span both clusters, each barrier waits on slower E-cores and cross-cluster cache traffic: (38 − 21) ms / ~700 levels ≈ 20–25 µs extra per level. This is the CPU-side counterpart of the GPU per-level sync cost that Phase 7 targets.
- *Weak scaling from 4 to 8 threads on LiveJournal* (dynamic: 184 → 141 ms, 1.3×): adding E-cores helps, but much less than adding P-cores did (1 → 4 threads: 3.0×).
- *One-thread cpu_omp is 1.3× slower than cpu_serial* (561 vs 425 ms on LiveJournal). That's the cost of the CAS claim plus buffer and reduction machinery. Real parallel speedup over serial at 8 threads is about 3×.
- *Direction optimization is the big algorithmic win:* 6.2× over cpu_omp on LiveJournal, 7.9× on Orkut, 6.2× on Pokec, 2.2× on web-Google, and nothing on roadNet-CA. On roadNet-CA the frontier never exceeds `edges_to_check/alpha`, so it never switches to bottom-up.
- *Tiny graphs:* on ego-Facebook (0.1 ms) serial wins outright, because OpenMP region overhead exceeds the work.
- Caveat: macOS cannot pin threads to cores. "≤4 threads run on P-cores" is the scheduler's usual behavior for default-QoS work, not something guaranteed.

**Decisions**
- **BFS sources come from the largest connected component** (degree > 0). This adds to PLAN 8. A roadNet-CA source in a 2-vertex component gave 1 edge / 0.34 ms, and that single source pulled the harmonic-mean GTEPS down to 0.0002. Graph500 implicitly avoids this, since Kronecker graphs keep nearly all non-isolated vertices in the giant component. `--any-component` restores the plain rule.
- **CSV schema:** the PLAN 8 columns plus a trailing `load1` (1-minute load average), which implements PLAN 8's "record uptime load average". `params` is space-separated with no commas (`dynamic,64` is written as `dynamic/64`).
- **The scaling sweep uses 16 sources × 3 trials** instead of 64 × 5. At 1 thread on LiveJournal the full protocol would take over an hour; sources are identical across configurations. The baseline table uses the full protocol.
- `gbench --trace` does one extra untimed, traced run per source, so the timed runs carry no tracing overhead.
- `cpu_serial` was added to `bfs_cpu.cpp` in Phase 1 and is now joined by the OpenMP variants (no layout change).

**Problems found and fixed**
- **gbench memory and swap.** The first version kept all 64 reference `dist` arrays (64 × 4n bytes: 786 MB on Orkut, 1.2 GB on LiveJournal). With other apps running, the machine was 4.7 GB into swap, and the Orkut validation slowed to a crawl (>12 min for one variant) while the timed runs stayed normal at ~200 ms. Now each source's reference is computed just before its trials and then dropped. Orkut peak RSS is ~965 MB, and a full cpu_omp run takes 114 s. The Orkut rows were rerun with the fixed binary. The other datasets' rows came from the earlier binary; its timed code path is identical, and the LiveJournal numbers agree with the separate scaling run (138.7 vs 140.8 ms).

**Open issues / notes**
- System swap was ~4.7 GB used during this phase because of other apps (VS Code, browser). Timed runs looked stable, but closing other apps before the Phase 10 full run would help.

### 2026-10-15 · Phase 3: microbenchmarks

**Built**
- `gmicro --bandwidth`: sequential read, write, and copy kernels (`bw_read`, `bw_write`, `bw_copy` in `micro.metal`) plus OpenMP equivalents over the **same shared buffers**, for sizes 1 MB → 2 GB. Output: `results/micro_bandwidth.csv`.
- `gmicro --launch`: 1000 empty dispatches in four modes (below). Output: `results/micro_launch.csv`.
- `gmicro --random`: gathers `table[idx[i]]` with sequential and random `idx`, GPU and CPU, for table sizes 1 MB → 2 GB with a fixed 32M reads (128 MB idx). Output: `results/micro_random.csv`.
- `scripts/micro.sh` runs all three. Peak RSS is 2.16 GB for bandwidth and 2.29 GB for random, under the 3 GB limit.
- **Every run is self-checking:** data is `uint`, so sums wrap mod 2^32 and are exact in any order. The GPU read sum, the GPU/CPU gather sums (against a serial reference), the write pattern, and copy (memcmp) are all verified. All passed.

**Sequential bandwidth** (median GB/s of 5 trials; copy counts 2×size of traffic)

| size | GPU read | GPU write | GPU copy | CPU read | CPU write | CPU copy |
|---|---|---|---|---|---|---|
| 1 MB | 292 | 380 | 414 | 25 | 41 | 73 |
| 2 MB | 323 | 386 | 199 | 36 | 43 | 95 |
| 4 MB | 129 | 131 | 95 | 55 | 67 | 98 |
| 16 MB | 105 | 122 | 93 | 79 | 69 | 90 |
| 64 MB | 102 | 101 | 91 | 61 | 65 | 74 |
| 256 MB | 99 | 92 | 87 | 61 | 65 | 75 |
| 1 GB | 95 | 96 | 89 | 63 | 64 | 72 |
| 2 GB | 94 | 97 | – | 62 | 64 | – |

- **GPU DRAM bandwidth ≈ 95 GB/s** (read, ≥256 MB), which is ~93% of the 102.4 GB/s theoretical for LPDDR5-6400 × 128 bit. This is the reference "peak sequential bandwidth" for later phases.
- **Cache regime:** at ≤2 MB the buffer stays in the GPU's on-chip caches across the repeated dispatches (290–410 GB/s). From 4–16 MB, 105–130 GB/s is the system-level cache (8 MB on M2) partly helping; at ≥32 MB it's DRAM. The ≤2 MB numbers jump around between runs (e.g. 2 MB copy 175–199) because these are tiny, cache-resident transfers. Treat them as order-of-magnitude only.
- **CPU (8 threads):** ~61 GB/s read, ~65 write, ~74 copy at DRAM sizes. One core reaches 29 GB/s read and 66 GB/s write, and 4 threads already saturate read (61.4 GB/s at 256 MB). The CPU side cannot use the full DRAM bandwidth; the GPU gets ~1.55× more.
- CPU numbers at 1–2 MB are low because each rep is one OpenMP parallel region: the ~20 µs fork/join plus barrier (see Phase 2) is comparable to the transfer itself.

**Kernel launch cost** (1000 empty dispatches, 1 threadgroup of 32, median of 5 trials)

| mode | wall µs / dispatch | GPU µs / dispatch | notes |
|---|---|---|---|
| `sync_each` (new command buffer, commit + `waitUntilCompleted` each) | 113.4 | 6.7 | median 111 µs, p90 117 µs per round trip |
| `async_each` (new command buffer each, wait on last) | 29.1 | 28.9 | submission throughput, no round trip |
| `encoder_each` (one command buffer, new encoder each) | 1.1 | 0.6 | |
| `one_encoder` (one command buffer, one encoder) | 0.8 | 0.6 | dependent dispatches, serial dispatch type |

- **A host round trip costs ~111 µs**, and only ~7 µs of that is GPU-side. The rest is commit → schedule → completion → wake-up of the waiting CPU thread. A level-synchronous GPU BFS that reads the frontier size on the host every level pays this per level. On roadNet-CA (~700 levels) that is ~78 ms of pure overhead, about 2× the *whole* serial CPU BFS (37.5 ms). Phase 7's hybrid targets exactly this.
- Dependent dispatches inside one encoder cost only ~0.6–0.8 µs each, ~140× cheaper. Keeping consecutive levels on the GPU without host readback (indirect dispatch, the Phase 7 stretch goal) is therefore very attractive.
- Even without waiting, each separate command buffer costs ~29 µs: batch work into as few command buffers as possible.

**Random gather vs sequential** (median Greads/s; ratio = sequential / random)

| table | GPU seq | GPU rand | ratio | CPU seq | CPU rand | ratio |
|---|---|---|---|---|---|---|
| 1 MB | 19.2 | 9.03 | 2.1× | 12.7 | 5.16 | 2.5× |
| 4 MB | 16.2 | 2.69 | 6.0× | 11.6 | 5.28 | 2.2× |
| 16 MB | 16.8 | 1.09 | 15.4× | 10.4 | 1.58 | 6.6× |
| 64 MB | 16.2 | 0.54 | 30× | 8.5 | 0.39 | 22× |
| 256 MB | 12.2 | 0.31 | 39× | 7.0 | 0.34 | 21× |
| 2 GB | 9.8 | 0.27 | 36× | 6.0 | 0.32 | 19× |

- **Random 4-byte gathers from DRAM-sized tables run at ~0.27–0.37 Greads/s on the GPU**, about 1.1–1.5 GB/s of useful payload and 35–45× slower than sequential. Every read pulls a whole cache line, so the payload fraction is tiny. This is the graph access pattern (`dist[col_idx[e]]`, `contrib[col_idx[e]]`), and it, not the 95 GB/s, is what bounds BFS and PageRank on large graphs.
- GPU sequential gather at DRAM sizes is 12 Greads/s × 8 B (idx + value) ≈ 97 GB/s, i.e. DRAM-bound as expected.
- **Surprise: for DRAM-sized tables the 8-thread CPU gathers randomly about as fast as the GPU** (0.32–0.35 vs 0.27–0.37 Greads/s). The GPU's advantage is in streaming (≈1.6×) and in cache-resident random access (≤16 MB: up to 1.8×), not in DRAM latency-bound access. Increasing parallelism didn't help: grids of 64K, 1M, and 16M threads give the same ~0.30 Greads/s at 256 MB.
- The CPU's random-gather cliff falls between 8 MB and 32 MB (4.5 → 0.5 Greads/s), consistent with the 16 MB P-cluster L2 plus 8 MB SLC. The GPU's falls earlier, between 2 MB and 4 MB.
- **Unexplained, reproducible:** GPU random gather is non-monotonic between 8 and 32 MB. 32 MB (1.6–1.9 Greads/s) beats 8 MB and 16 MB (0.75–1.6), in every run. A buffer-mapping effect is **ruled out**: allocating 32 or 256 MB but using only 8 MB (`--min-alloc-mb`) gives the same slow numbers. I'm noting it and not chasing it further, since it doesn't affect the DRAM-regime reference numbers.

**Implications recorded for later phases**
- **Phase 7 prediction (preliminary):** a GPU level costs ≥ ~111 µs round trip; a CPU level costs ~20 µs barrier + work. With CPU random gathers at ~0.3–1.5 Greads/s (DRAM to cache-resident), the CPU finishes a level in the ~90 µs difference if it touches fewer than roughly **3×10⁴ – 10⁵ edges**. So `T` should land around 10⁴·⁵–10⁵ edges. To be checked against the sweep.
- **Phase 8:** pull PageRank streams `row_ptr` + `col_idx` at near-sequential rates but gathers `contrib` randomly. Report effective bandwidth against both 95 GB/s (sequential) and ~0.3 Greads/s (random), as PLAN says. Reordering (Phase 9) should help by moving gathers toward the cache-resident regime.

**Decisions**
- **"Size" means bytes per buffer.** Copy needs two buffers, so it stops at 1 GB per buffer (2 GB resident) to respect the 3 GB process limit (`--copy-max-mb`). Read and write go to 2 GB.
- **Batched timing:** each trial encodes enough back-to-back dispatches in one command buffer to move ~1 GB (`--target-mb`), and GPU time comes from `GPUStartTime`/`GPUEndTime`. Without batching, small sizes would measure the 111 µs round trip instead of bandwidth.
- **Grid-stride kernels with a 16M-thread grid cap** (`--grid`). A 64K grid was fine for read but **halved copy bandwidth** (41 vs ~90 GB/s at 512 MB, consistent over repeats), and ≥1M was needed for write and copy. 16M means 1–8 uint4 elements per thread and was best or tied everywhere.
- **Time-based warm-up** (repeat until ≥200 ms, at least once) for every GPU and CPU measurement. With a single warm-up run, small GPU measurements ramped up to 2× across the timed trials (GPU clock ramp-up). After the fix, trials agree within ~5%.
- **CPU read uses 16 independent accumulators.** A plain `s += a[i]` reduction compiled to scalar adds and capped one core at 11.8 GB/s (compute-bound), which made the CPU baseline unfair. With 16 lanes: 29 GB/s per core, 61 GB/s at 8 threads.
- The read kernels reduce to one value per threadgroup (simd_sum + threadgroup reduction) instead of one per thread. This keeps loads live with negligible extra write traffic.

**Open issues / notes**
- System swap was still ~4.6 GB used, with other apps open. Results were stable across repeats, but the Phase 10 rerun should be done with other apps closed.

### 2026-10-15 · Phase 4: Metal infrastructure + first GPU BFS

**Built**
- `src/shaders/common.h`: `tg_sum` / `tg_max` threadgroup reductions (simd_sum, then a cross-simdgroup step), shared by every kernel file.
- `primitives.metal` + `src/metal/primitives.{hpp,mm}` (`gpu::Primitives`). Each operation *encodes* into the caller's compute encoder, so later phases can chain primitives with BFS kernels in one command buffer:
  - `fill`
  - `reduceSum` / `reduceMax`: two fixed passes, ≤1024 partials, then one 1024-thread group. Deterministic, no atomics.
  - `scanExclusive`: three-phase. Blocks of 1024 (256 threads × 4 items, simd_prefix_exclusive_sum), recursive scan of block sums, add-back. Works in place, and optionally writes the grand total.
  - `queueToBitmap` (atomic_fetch_or) / `bitmapToQueue` (ballot-style aggregated append: one atomic per simdgroup).
  - Scratch buffers are pooled, because allocation inside timed loops is too slow.
- `bfs.metal` + `src/metal/bfs_gpu.{hpp,mm}` (`gpu::GpuBfs`):
  - `gpu_vertex`: thread per vertex every level, with plain racy-but-benign dist writes (every writer stores the same level+1).
  - `gpu_naive`: thread per frontier vertex. It claims neighbors with a CAS on dist (with a **retry loop**: MSL only has the *weak* CAS, which may fail spuriously and would otherwise silently drop a vertex) and appends with one atomic per claim.
  - Both kernels aggregate level statistics per simdgroup (frontier size, edges examined, discovered): one atomic per 32 threads.
  - The host loop is one command buffer per level, with commit + wait, then a read of the stats from the shared buffer. Initialization rides in level 0's command buffer.
- **Zero-copy graph:** `GpuBfs` copies the CSR once into shared buffers and exposes a CSR *view* of them (`graph()`). gbench then drops its host copy and runs even the CPU reference on the GPU's bytes, so the graph is resident once. GPU runs write into a shared `dist` buffer that gbench validates in place, with no copy-out.
- gbench: `gpu_vertex` / `gpu_naive`, `--tg` (threadgroup size), `time_ms_gpu` column (sum of GPUStart→End over the run's command buffers), and `--source-warmup` (below). It is now compiled as Objective-C++ (still `main.cpp`, via a Makefile rule).
- `scripts/bench_bfs.sh CSV VARIANTS...`: every dataset with the full protocol and `--validate`. Optional `TRACE=` saves per-level traces for LiveJournal and roadNet-CA (4 sources).
- **Tests: 18/18** (5 new).
  - Primitives are tested at sizes 0, 1, 2, 31–33, 255–257, 1023–1025, 4097, 100003, 2^20 ± 1, and 3000017 (one-, two- and three-level scans), plus in-place scan, fill with offset, and the queue→bitmap→queue round trip.
  - Both GPU BFS variants run on every synthetic graph with 7 sources (including isolated vertices and the max-degree hub) at tg 32/256/1024. They must match exactly, and the trace must count every reached vertex and every edge exactly once.
  - Mutation check: deliberately breaking the scan (off by one at one element) or replacing the CAS claim with a racy load/store makes 3 tests fail, so the tests do catch these bugs.

**Validation:** both variants match `cpu_serial` exactly on all six datasets × 64 sources × every trial (`--validate`). Peak RSS for the whole run is 1.94 GB (Orkut).

**First GPU numbers** (`results/bfs_gpu.csv`; rerun 2026-10-19 on AC, Low Power Mode off; 64 sources × 5 trials, validated; median ms end-to-end / kernel-only; GTEPS = harmonic mean). Before the run, two reference points matched the earlier AC measurements: `cpu_do` LiveJournal 23.3 ms (Phase 2: 22.4) and launch round trip 104 µs (Phase 3: 111). Peak RSS 1.45 GB.

| dataset | gpu_vertex e2e / GPU | GTEPS | gpu_naive e2e / GPU | GTEPS | CPU (Phase 2): omp / do / serial |
|---|---|---|---|---|---|
| ego-Facebook | 2.6 / 1.4 | 0.035 | 4.4 / 3.2 | 0.021 | – / – / 0.10 |
| web-Google | 14.0 / 11.0 | 0.30 | 17.2 / 14.4 | 0.25 | 15.2 / 6.9 / 32.6 |
| roadNet-CA | 320 / 190 | 0.0085 | 115 / 19.6 | 0.024 | 39.3 / 41.3 / 37.5 |
| soc-Pokec | 17.8 / 15.9 | 1.25 | 29.8 / 27.4 | 0.75 | 54.7 / 8.8 / 162 |
| soc-LiveJournal1 | 36.2 / 33.3 | 1.16 | 61.9 / 59.3 | 0.68 | 139 / 22.4 / 425 |
| com-Orkut | 75.6 / 73.7 | 1.53 | 119 / 117 | 0.99 | 222 / 28.0 / 555 |

- vs `cpu_omp` (same top-down algorithm, 8 threads): `gpu_vertex` is **3.8× faster on LiveJournal, 2.9× on Orkut, 3.1× on Pokec**.
- vs `cpu_do`: `gpu_vertex` is still **1.6× slower on LiveJournal, 2.7× on Orkut, 2.0× on Pokec**.
- roadNet-CA: the best GPU variant (115 ms) is **3.1× slower than serial CPU** (37.5 ms).
- The earlier battery/Low Power Mode runs (Orkut `gpu_vertex` 102–164 ms vs 75.6 now) are superseded. They remain only in git history (`445fb3e`).

**Observations**
- **Both simple GPU variants beat `cpu_omp` on the social graphs** (LiveJournal ~1.5–3×), but **lose clearly to `cpu_do`**. Top-down BFS does ~m edge checks regardless of device; direction optimization skips most of them. `gpu_do` (Phase 6) is the fair comparison.
- **roadNet-CA is the sync-bound case, as Phase 3 predicted.** `gpu_naive` runs ~608 levels; the GPU works for only 19.6 ms, but end to end takes 115 ms, i.e. ~0.19 ms per level, of which ~0.03 ms is kernel. The rest is the ~104–111 µs round trip plus per-level command-buffer setup. `gpu_vertex` is worse still (190 ms of kernel time), because it launches all 2M vertex threads on every one of the 608 levels just to find the few frontier vertices.
- **`gpu_naive` is slower than `gpu_vertex` on every social graph.** The LiveJournal trace shows where: the big levels 3–5 (2.3M–39M edges) take 16.6 / 20.7 / 26.8 ms with `gpu_naive` vs 8.2 / 13.3 / 11.6 ms with `gpu_vertex`. There are two causes:
  - Load imbalance: one thread walks a whole hub adjacency list (max degree ~20K) while its simdgroup waits.
  - One global atomic per discovered vertex on a single queue-tail counter (2.4M at level 5).

  `gpu_vertex` avoids both: its threads are mostly idle filters, and it has no queue. On tiny frontiers (late levels), `gpu_naive` wins (median roadNet-CA level 0.16 vs 0.47 ms), because it launches only `size` threads. Phase 5 (`gpu_binned` / `gpu_edge`) targets the imbalance and contention.
- **GPU clock wake-up:** in traces, level 0 of the first source took 16–24 ms (vs ~0.4 ms for the other sources) after the GPU sat idle during the CPU reference computation.

**Decisions**
- **`--source-warmup 1` (default, all variants):** one untimed run per source after computing its reference and before its timed trials. Without it, trial 0 of every source was ~15% slower, because the serial reference (up to ~0.5 s) let the GPU clock down; with it, LiveJournal trial medians are flat (31.3–32.0 ms). This adds to PLAN 8's "2 warm-up runs". The Phase 2 CPU baseline predates it; Phase 10 reruns everything with it.
- **GPU timed region = everything per run:** init kernel, every level, every host sync, and reading the stats. Upload and graph transforms are excluded (PLAN 4.4). `time_ms_gpu` is reported next to it.
- **The graph is uploaded by copy** (`newBufferWithBytes`). Then the host copy is dropped, except for undirected inputs, where the mmap'd stored graph stays mapped in gbench. Its pages are clean and file-backed, so the OS can reclaim them; peak RSS is 1.94 GB. `newBufferWithBytesNoCopy` on the mmap is still not used, because the 32-byte header leaves the arrays non-page-aligned. Upload is outside timing, so this only matters for memory.
- `gpu_vertex` uses plain loads/stores on dist (a benign race). `STAT_FOUND` can overcount there, so it serves only as the "anything discovered?" flag. Exact frontier sizes come from the next level's frontier count.

**Open issues / notes**
- Swap was ~6.8 GB used during the AC rerun (other apps open). Timings were stable (median within ~5–15% of min on social graphs), but close apps before Phase 10.

### 2026-10-19 · Phase 5: load-balanced GPU BFS

**Built**
- **`gpu_binned`** (`bfs_binned_small/medium/large`): three degree bins, three dispatches per level.
  - small (< `--small-max`): simdgroup-cooperative gather (see below)
  - medium: simdgroup per vertex, lanes stride the list
  - large (≥ `--large-min`): threadgroup per vertex
- **`gpu_edge`** (`bfs_edge_expand`): exclusive scan of frontier degrees (the Phase 4 scan primitive), then a fixed `--ept` edges per thread.
  - Threadgroup *t* owns a contiguous chunk of tg × ept edges; thread *j* takes edges *j*, *j*+tg, ..., so col_idx reads are coalesced.
  - Thread 0 binary-searches the chunk's first and last owner once; each edge then searches only that range (usually 1–2 vertices on hub chunks).
- **Still one host sync per level** (Phase 3: ~111 µs each). All per-vertex bookkeeping happens at discovery time in the previous level, and the host learns everything it needs from the same sync:
  - `gpu_binned` appends a discovered vertex directly to the next queue of its degree bin, so the host reads three bin counts.
  - `gpu_edge` stores each discovered vertex's degree next to its queue slot (scanned in place next level), and the level's total edge count accumulates in `stats[3]`.
  - The rejected alternative, a classification pass, needs a second sync per level to size its dispatches.
- **Aggregated queue appends:** one atomic per simdgroup per round (prefix sum + broadcast) instead of one per discovered vertex. This requires simdgroup-uniform loop trip counts, so all append loops run uniform rounds with a per-lane `has` flag. A `simd_any(claimed)` guard skips the append work for rounds that discover nothing. That guard is cheap and correct, but it made no measurable difference; see the bottleneck analysis.
- `scripts/sweep_balanced.sh` → `results/sweep_bfs_balanced.csv`. `results/bfs_star.csv` holds every variant on `synth:star100k`.
- **Tests: still 18, now covering 10 GPU configurations** per graph × source × tg ∈ {32, 256, 1024}:
  - `gpu_binned` with default, all-small, all-large, all-medium and 2/3 thresholds
  - `gpu_edge` with ept 1/4/16
- **Mutation checks (each caught by the tests):**
  - owner binary search `<=` → `<`
  - aggregated append slot off by one
  - small-bin loop made divergent (per-lane trip count instead of simdgroup max). This confirms the uniform-rounds requirement is real.
  - cooperative owner search `<=` → `<`

**Small-bin redesign (measured).** The first small kernel had each lane walk its own list for max-degree-in-simdgroup rounds. On LiveJournal, putting every vertex into the small bin took 86.7 ms vs 42.3 ms for all-medium, and the default thresholds gave 61.6 ms, *slower than `gpu_vertex`*. Reads were uncoalesced, and short-list lanes idled.

The replacement is Merrill's warp-cooperative gather. The 32 short lists are concatenated via `simd_prefix_exclusive_sum` of degrees; each round, lane *j* takes edge *j* and finds its owner with a 5-step `simd_shuffle` binary search. All-small dropped to 48.8 ms and the default to 47.5 ms. The bin's contract is unchanged (one thread owns each small vertex); only the edge expansion is shared.

Two other hypotheses tested and **rejected** (no measurable change on LiveJournal):
- plain instead of atomic loads for the "already visited?" filter before the CAS
- skipping append work when nothing was claimed

**Tuning** (`results/sweep_bfs_balanced.csv`; 16 sources × 3 trials, median e2e ms)
- **`gpu_binned`:** 12 threshold pairs vary within ~10% on LiveJournal (46.0–54.8) and Orkut (66.2–71.6). That's near the ~5% run-to-run noise, because the cooperative small bin removed most of the sensitivity. Threadgroup size 64–1024 also varies within noise. Defaults: **small < 16, large ≥ 256, tg 256** (LiveJournal 47.5, Orkut 66.2, best on Orkut).
- **`gpu_edge`:** **ept 2** is best on both graphs at every tg (LiveJournal 38.6–39.9, Orkut 64.6–65.4, except tg 1024 on Orkut). ept 1 pays the owner search and append overhead on every edge (Orkut 74–85). ept ≥ 8 cuts parallelism on mid-size levels (Orkut 76–94). Default: **ept 2, tg 256**.

**Results** (`results/bfs_gpu.csv`; all four variants in one session on AC power, Low Power Mode off; 64 sources × 5 trials, all validated; median ms end-to-end, GPU-only in parentheses; peak RSS 1.94 GB)

| dataset | gpu_vertex | gpu_naive | gpu_binned | gpu_edge | best CPU (Phase 2) |
|---|---|---|---|---|---|
| ego-Facebook | 2.7 (1.4) | 4.4 (3.2) | 1.6 (0.39) | **1.2** (0.21) | serial 0.10 |
| web-Google | 16.7 (11.8) | 17.3 (14.4) | **10.9** (7.5) | **10.9** (6.9) | do 6.9 |
| roadNet-CA | 368 (233) | **115** (29) | 119 (35) | 115 (27) | serial 37.5 |
| soc-Pokec | 18.6 (16.6) | 30.1 (27.9) | 18.5 (16.7) | **18.0** (16.0) | do 8.8 |
| soc-LiveJournal1 | **36.8** (34.1) | 66.8 (63.5) | 44.3 (41.1) | 42.9 (40.0) | do 22.4 |
| com-Orkut | 81.5 (79.7) | 131.9 (129.3) | **68.6** (66.8) | 77.5 (75.5) | do 28.0 |

**Star graph** (`synth:star100k`, 64 leaf sources; `results/bfs_star.csv`): `gpu_naive` takes **41.4 ms**, `gpu_vertex` 15.2, `gpu_binned` 1.05, **`gpu_edge` 0.67 ms, 62× faster than naive**. The single hub's 100K-edge list is walked by one thread in `gpu_naive`, by one threadgroup (one core) in `gpu_binned`, and by the whole GPU in `gpu_edge`. The CPU wins anyway (serial 0.16 ms): the graph is tiny, and each GPU level costs a round trip.

**Interpretation**
- **Load balancing fixes `gpu_naive`:** 1.5–1.9× faster on the social graphs (Orkut 132 → 69 ms, LiveJournal 67 → 43, Pokec 30 → 18) and 40–60× on the star.
- **But balanced variants barely beat the simple `gpu_vertex`, and lose to it on LiveJournal.** Top-down BFS on these graphs is **bound by random `dist` gathers, not by imbalance**. LiveJournal examines ~85M adjacency entries per BFS, each a 4-byte random read of a 19 MB `dist` array. 85M / ~34–41 ms ≈ 2.1–2.5 G reads/s. That is at, or a bit above, the Phase 3 random-gather rate for a table that size (1.1–1.9 G/s at 16–32 MB; real graphs have some locality). Balancing changes who does the gathers, not how many there are. Only skipping edges (direction optimization, Phase 6) or improving locality (reordering, Phase 9) attacks that.
- **The per-level trace shows both effects** (LiveJournal, ms per level):

  | level | edges | gpu_naive | gpu_vertex | gpu_binned | gpu_edge |
  |---|---|---|---|---|---|
  | 3 | 2.3M | 16.7 | 8.6 | 2.6 | 2.2 |
  | 4 | 34M | 25.3 | 24.5 | 24.9 | 26.1 |
  | 5 | 39M | 29.3 | 13.0 | 23.4 | 21.7 |
  | 6 | 8.7M | 8.1 | 2.5 | 7.6 | 6.9 |

  - **Level 3** (23K frontier vertices with hubs): load balancing gives a 4–8× speedup.
  - **Levels 4–5:** everything converges to the gather limit.
  - **Open question:** `gpu_vertex` is ~2× faster on levels 5–6, where the frontier is half the graph and most neighbors are already visited. My hypothesis is access order: `gpu_vertex` walks frontier vertices in vertex-ID order, so its `col_idx` reads stream sequentially and its `dist` reads keep the input ordering's locality, while queue variants process vertices in atomic-append order (effectively random). Not verified. Phase 6's bitmap bottom-up processes vertices in ID order and will show whether the effect holds; Phase 9 measures ordering effects directly.
- **roadNet-CA is unchanged** (~115 ms for every queue variant): ~600 levels × the per-level round trip, with only 27–35 ms of GPU work. Load balancing cannot help here; this is Phase 7's target.
- **Tiny graphs:** `gpu_edge` is best on ego-Facebook (1.2 ms), but CPU serial is 12× faster still.

**Decisions**
- The small bin uses a simdgroup-cooperative gather instead of a literal "each thread walks its list" loop (measured above).
- Defaults: `gpu_binned` small < 16, large ≥ 256; `gpu_edge` ept 2; tg 256 for both.
- The sweep uses the reduced protocol (16 sources × 3 trials), as in Phase 2; the results table uses the full protocol.

**Open issues / notes**
- The `gpu_vertex` level-5/6 advantage (access-order hypothesis) is to be checked in Phase 6/9.
- The three bin dispatches run serially in one encoder. They are independent within a level, so a concurrent dispatch type could overlap the small-bin tail with large-bin work. That's a possible later optimization; not needed now, since levels are gather-bound.

### 2026-10-19 · Phase 6: direction-optimizing GPU BFS

**Built**
- **`gpu_do`** (`GpuBfs::runDO`):
  - **Top-down levels** use `gpu_edge`, which was the best top-down variant on LiveJournal, Pokec and the star graph. Its kernel encoding was factored out into `encodeEdgeLevel`.
  - **Bottom-up levels** use `bfs_bottom_up`: one thread per vertex. An unvisited vertex scans its neighbors and **exits at the first frontier hit** in a bitmap. The bitmap is n/8 bytes (600 KB for LiveJournal) and stays in cache, so the random reads no longer go to the 19 MB `dist` array. No CAS is needed, since each vertex's own thread is the only writer of its `dist`.
  - **Next-frontier bitmap without atomics:** a simdgroup's 32 lanes are 32 consecutive vertices, i.e. exactly one bitmap word. `simd_or` assembles the word and lane 0 stores it. Every word is overwritten, so no clearing pass is needed.
- **Switching rule is identical to `cpu_do`** (GAP), including `scout = 1` after a bottom-up phase. `m_f` comes from `stats[3]`, the degree sum of newly discovered vertices, which both the top-down and bottom-up kernels accumulate. `m_u` is tracked on the host.
- **Still one command buffer and one sync per level.** Conversions are encoded at the start of the first level in the new direction:
  - top-down → bottom-up: `queueToBitmap`
  - bottom-up → top-down: `bitmapToQueue`, plus a new `bfs_queue_degrees` kernel that rebuilds `gpu_edge`'s scan input
- **Tests: 19/19.** New: `gpu_do_matches_serial_and_cpu_decisions`. On every synthetic graph × 7 sources × 4 modes (default; forced bottom-up; never bottom-up; α = β = 2 flip-flop) × tg 32/256, the test checks:
  - exact `dist`
  - every reached vertex counted once
  - **the same direction and frontier size at every level as `cpu_do`**
- **Mutation checks (each caught):** wrong bit position in the `simd_or` word; corrupted degree rebuild after switching back; switching rule `>=` → `>` (caught by the CPU/GPU decision comparison).
- `scripts/sweep_do.sh` → `results/sweep_bfs_do.csv`. Results: `results/bfs_gpu_do.csv`. Traces (4 sources each, LiveJournal + roadNet-CA): `results/trace_bfs_gpu_do.csv`.

**α/β sweep** (`results/sweep_bfs_do.csv`; 16 sources × 3 trials, median e2e ms)

| α \ β | LJ 10 | LJ 18 | LJ 24 | Pokec 10 | Pokec 18 | Pokec 24 |
|---|---|---|---|---|---|---|
| 5 | 13.2 | 12.7 | 12.1 | 6.15 | **5.89** | 6.09 |
| 10 | 11.3 | 9.90 | 10.6 | 6.32 | 6.21 | 6.20 |
| 15 | 9.94 | **9.68** | 9.78 | 6.20 | 6.12 | **5.89** |
| 20 | 10.4 | 10.1 | 10.4 | 5.98 | 6.32 | 6.16 |
| 30 | 11.2 | 10.6 | 10.9 | 6.59 | 6.35 | 6.91 |

**Defaults: α = 15, β = 18** (GAP's values). They are best on LiveJournal and within run-to-run noise (~5%) of the best on Pokec, where all pairs span only 5.9–6.9 ms. Switching too early (α = 5) costs ~30% on LiveJournal: bottom-up starts while the frontier is still small, and unvisited vertices scan long lists without hits. Switching late (α = 30) costs ~10%.

**Results** (`results/bfs_gpu_do.csv`; 64 sources × 5 trials, validated; median ms e2e, GPU-only in parentheses; peak RSS 1.6 GB). **Power: battery, Low Power Mode off.** The charger was unplugged during this phase. Before the run, I checked the GPU-side reference points against AC: launch round trip 112 µs (AC 104–111) and `gpu_edge` LiveJournal 38.8 ms (Phase 5 AC sweep 38.7), so GPU numbers look unaffected. CPU `cpu_do` was ~17% slower on battery (27.1 vs 22.4–23.3), which is why the CPU column below uses the Phase 2 AC numbers.

| dataset | gpu_do | best top-down GPU (Phase 5) | speedup | cpu_do (Phase 2, AC) | gpu_do vs cpu_do |
|---|---|---|---|---|---|
| ego-Facebook | 1.56 (0.37) | gpu_edge 1.20 | 0.77× | 0.38 (serial 0.10) | slower |
| web-Google | 9.2 (6.4) | binned/edge 10.9 | 1.18× | 6.9 | 0.75× |
| roadNet-CA | 133 (30) | naive/edge 115 | ~0.9× (no bottom-up) | 41.3 (serial 37.5) | slower |
| soc-Pokec | **6.0** (4.1) | gpu_edge 18.0 | **3.0×** | 8.8 | **1.5×** |
| soc-LiveJournal1 | **11.1** (8.5) | gpu_vertex 36.8 | **3.3×** | 22.4 | **2.0×** |
| com-Orkut | **13.8** (12.2) | gpu_binned 68.6 | **5.0×** | 28.0 | **2.0×** |

- **On the social graphs, `gpu_do` is 3.0–5.0× faster than the best top-down GPU variant and 1.5–2.0× faster than `cpu_do` (8 threads).** This is the first GPU variant that beats the best CPU code. Orkut reaches 7.5 GTEPS (harmonic mean).
- The Orkut median (13.8) vs min (7.1) spread is unusually wide; possibly battery-related. Revisit on AC in Phase 10.
- **roadNet-CA never goes bottom-up** (0 bottom-up levels over 4 traced sources × 774 levels, the same as `cpu_do` in Phase 2), so `gpu_do` is just `gpu_edge` plus a check. It's still sync-bound: 133 ms e2e vs 30 ms GPU.
- web-Google gains only 1.18×. It's a small graph where many levels are tiny and sync costs dominate.
- ego-Facebook: the extra bottom-up machinery doesn't pay off on 4K vertices.

**LiveJournal trace** (one source; direction per level `TTTTBBBTTTTTTT`)

| level | frontier | edges examined | dir | ms | top-down `gpu_edge` (Phase 5) |
|---|---|---|---|---|---|
| 3 | 23K | 2.3M | T | 2.2 | 2.2 |
| 4 | 607K | **14.0M** | B | 7.0 | 26.1 (34.4M edges) |
| 5 | 2.42M | **2.6M** | B | 2.4 | 21.7 (39.3M edges) |
| 6 | 1.49M | 0.39M | B | 1.5 | 6.9 (8.7M edges) |
| 7 | 265K | 0.86M | T | 1.6 | – |

The three bottom-up levels examine **17M adjacency entries instead of 82M** (early exit). Their random reads hit the cache-resident bitmap, so the levels that took ~55 ms top-down take ~11 ms. Levels 8–13 are tiny and cost ~0.2 ms each, which is basically the sync. The 23 ms for level 0 in the trace is the GPU wake-up on the first traced source, seen before in Phase 4.

**Phase 5 open question (gpu_vertex faster on levels 5–6):** not settled. Bottom-up also walks vertices in ID order, but it does far less work per level, so the comparison doesn't isolate access order. Left for Phase 9 (orderings).

**Decisions**
- Top-down inside `gpu_do` = `gpu_edge` (not `gpu_binned`). It was best on LiveJournal and Pokec; `gpu_binned` was better on Orkut (68.6 vs 77.5), but Orkut's big levels go bottom-up anyway.
- α = 15, β = 18 kept (sweep above).
- `--alpha` / `--beta` are shared CLI options for `cpu_do` and `gpu_do`.

**Open issues / notes**
- The `gpu_do` table was measured on battery (GPU reference points matched AC). Rerun on AC in Phase 10 with everything else.
- The remaining cost on the social graphs is mostly per-level syncs on the ~7–10 small levels (~0.2 ms each, ≈2 ms of 11 on LiveJournal) and the big bottom-up level. That's Phase 7's territory.

### 2026-10-19 · Phase 7: hybrid CPU+GPU BFS

**Built**
- **`src/cpu/bfs_steps.{hpp,cpp}`**: single BFS levels on the CPU that use the **GPU's exact data layout** in the Metal shared buffers.
  - top-down: queue in → queue out plus degrees, CAS claims, per-thread buffers with one atomic append each
  - bottom-up: bitmap in → bitmap out, a thread owns whole words, no atomics
  - queue↔bitmap conversions and `dist` init
  - Each step returns the counters the GPU kernels put in their stats buffer, so any level can run on either device with **no copies**.
  - Levels under 2048 edges run on one thread, because an OpenMP region costs ~20 µs (Phase 2), more than the whole level.
- **`GpuBfs::runDirOpt(threshold)`:** one direction-optimizing state machine. `gpu_do` = threshold 0; `hybrid` = `--T`.
  - **Per level:** the CPU runs it if its work is below T. Work is the frontier's edges for a top-down level (known exactly), and the unexplored edges `m_u` for a bottom-up level (an upper bound, since bottom-up work is only known afterwards).
  - A GPU level is one command buffer plus a sync; **a CPU level needs no sync at all**.
  - Conversions run on whichever device executes the first level in the new direction.
  - Correctness relies on unified-memory coherence: CPU writes before a commit are visible to that command buffer, and GPU writes are visible to the CPU after `waitUntilCompleted`.
- gbench `hybrid`: `--T N|inf` (default 1e5), CPU threads `--threads` (default 4 for hybrid).
- `scripts/sweep_hybrid.sh` → `results/sweep_hybrid.csv`. Results: `results/bfs_hybrid.csv`. Traces: `results/trace_bfs_hybrid.csv`, LiveJournal + roadNet-CA × 4 sources at T = 1e5 *and* T = 0, with a device column.
- **Tests: 20/20.** New: `hybrid_matches_serial_at_every_threshold`. On every synthetic graph × 7 sources × 3 heuristic modes × T ∈ {0, 1, 64, 1000, 1e5, ∞} × 1/8 CPU threads, it checks:
  - exact `dist`
  - **the same direction and frontier at every level as `cpu_do`** (the device choice must not change the algorithm)
  - T = 0 never uses the CPU, T = ∞ never the GPU, and a top-down level 0 is on the CPU iff degree(source) < T
- **Mutation check:** making the CPU step write wrong degrees for the GPU's next scan fails only mixed-device runs (grid100x100, T = 64). That confirms the test exercises the hand-off.

**Prediction (written before the sweep).**
- GPU level: ~0.16–0.19 ms fixed for tiny top-down levels (Phase 4/5 traces: the ~111 µs Phase 3 round trip plus command-buffer setup and scan dispatches), then ~2 G edges/s.
- CPU level: ~20 µs fork/join plus ~0.62 G edges/s (`cpu_omp` LiveJournal, Phase 2), serial below 2048 edges.
- Break-even: 20 µs + E/0.62G = 190 µs + E/2G gives **E ≈ 1.5×10⁵**. Phase 3's rougher estimate was 3×10⁴–10⁵. **Predicted best T ≈ 10⁵**, with roadNet-CA nearly all-CPU.

**Sweep** (`results/sweep_hybrid.csv`; 16 sources × 3 trials; median e2e ms)

| T | roadNet 4 thr | roadNet 8 thr | LJ 4 thr | LJ 8 thr | web-Google 4 thr | web-Google 8 thr |
|---|---|---|---|---|---|---|
| 0 (all GPU) | 115.7 | 124.3 | 11.1 | 11.1 | 8.9 | 9.0 |
| 1e3 | 110.1 | 101.8 | 10.2 | 10.1 | 7.6 | 7.8 |
| 1e4 | 73.4 | 72.5 | **9.0** | 10.0 | 7.4 | 7.3 |
| 3e4 | 27.2 | 33.0 | 9.4 | **9.5** | 5.7 | 7.1 |
| **1e5** | 26.2 | 30.2 | 9.6 | 9.6 | **5.1** | 7.0 |
| 3e5 | **25.5** | 31.3 | 10.1 | 9.7 | 7.1 | 6.9 |
| 1e6 | 26.7 | 32.2 | 13.1 | 11.0 | 8.1 | 7.4 |
| 1e7 | 28.0 | 30.1 | 14.3 | 15.4 | 9.0 | 6.5 |
| ∞ (all CPU) | 27.8 | 31.6 | 31.5 | 23.4 | 8.5 | 6.7 |

**Result: the prediction holds.** T = 10⁵ is inside the best band on all three graphs (roadNet ≥ 3×10⁴; LiveJournal 10⁴–3×10⁵; web-Google best at 10⁵ with 4 threads). Defaults: **T = 10⁵, 4 CPU threads**. 4 threads beat 8 on roadNet-CA at every T ≥ 3×10⁴ (25–28 vs 30–33 ms), the same P/E-core barrier effect as in Phase 2. On LiveJournal they tie.

**Why the best T is what it is** (from the traces)
- **A GPU level has a fixed cost of ~0.17–0.24 ms** however small it is. roadNet-CA's GPU levels have a median of 0.198 ms; LiveJournal's tail levels with 1–4K edges take 0.17–0.24 ms. That is the round trip plus command-buffer and scan-dispatch setup. **A tiny CPU level costs 1–35 µs** (roadNet median 34 µs; LiveJournal tail levels 0.1–26 µs), because the serial path has no fork/join and the data is already in the shared buffers. So any level small enough to finish on the CPU in ~0.2 ms belongs there.
- **The CPU's top-down rate on mid-size levels is lower than I assumed:** ~0.11–0.2 G edges/s with 4 threads (LiveJournal level 3: 76K edges took 0.67 ms on the CPU vs 0.31 ms on the GPU), not the 0.62 G/s of 8 threads on huge levels. That moves the true break-even to **≈ 0.2 ms × 0.15 G/s ≈ 3×10⁴ edges**, a bit below my 1.5×10⁵ but still inside the Phase 3 range.
- **Why the curves are flat:** real graphs have few levels near the break-even. LiveJournal's levels jump from 76K edges (level 3) to 4.6M (level 4). Any T from ~10⁴ to ~4×10⁶ gives nearly the same device assignment, so the curve only rises when the big bottom-up levels (m_u ≈ 10⁷) move to the CPU (T ≥ 10⁶–10⁷). roadNet-CA's levels never exceed a few ×10⁴ edges, so once T ≥ 3×10⁴ everything is on the CPU and T stops mattering.
- **Rule of thumb:** T ≈ (GPU per-level fixed cost) × (CPU edge rate on mid-size levels) ≈ 0.2 ms × 0.1–0.6 G/s ≈ 2×10⁴–10⁵. This follows directly from Phase 3's ~111 µs round trip.

**Results** (`results/bfs_hybrid.csv`; T = 1e5, 4 CPU threads; 64 sources × 5 trials, validated; AC power, Low Power Mode off; median e2e ms; peak RSS 1.96 GB)

| dataset | hybrid | gpu_do (T=0, Phase 6) | best CPU (Phase 2) | hybrid vs gpu_do | hybrid vs best CPU |
|---|---|---|---|---|---|
| ego-Facebook | 0.95 | 1.56 | serial 0.10 | 1.6× | 0.1× |
| web-Google | 6.8 | 9.2 | do 6.9 | 1.4× | 1.0× |
| roadNet-CA | **25.6** | 133 | serial 37.5 | **5.2×** | **1.46×** |
| soc-Pokec | **5.0** | 6.0 | do 8.8 | 1.2× | **1.8×** |
| soc-LiveJournal1 | **9.3** | 11.1 | do 22.4 | 1.2× | **2.4×** |
| com-Orkut | **13.6** | 13.8 | do 28.0 | 1.0× | **2.1×** |
| star100k | 0.47 | 0.61 | serial 0.16 | 1.3× | 0.3× |

- **Hybrid is the best or tied-best non-trivial variant on every dataset except the two tiny graphs.** roadNet-CA, the case every GPU variant lost, is now **1.46× faster than serial CPU** and 5.2× faster than all-GPU.
- **Honest caveat:** most of roadNet's gain is from running its tiny levels serially on the CPU. All-CPU hybrid (T = ∞) is 27.8 ms, also faster than Phase 2's `cpu_do` (41.3), which uses an OpenMP region on every level. So on roadNet-CA "hybrid beats CPU" really means "the CPU path with a serial cutoff beats the plain OpenMP baseline". The GPU contributes nothing there. On the social graphs, both devices contribute: the GPU takes the big levels and the CPU removes ~10 syncs of ~0.2 ms.
- Orkut gains little, because its levels are few and mostly large.

**Stretch goal (indirect-dispatch multi-level GPU BFS): not done.** With hybrid, the remaining GPU levels are the big ones (≥ 0.66 ms each on LiveJournal), where the ~0.11 ms sync is ≤ 15% of the level. Keeping the tiny levels on the GPU via indirect dispatch would save ~0.17 ms per tiny level, the same as moving them to the CPU already does. It would matter for graphs with many mid-size levels; none of ours have that profile.

**Decisions**
- `hybrid` defaults: T = 1e5, 4 CPU threads. `gpu_do` is unchanged and is now `runDirOpt` with threshold 0.
- The work estimate for bottom-up levels is `m_u` (unexplored edges), so bottom-up levels go to the CPU only for very large T.
- CPU steps go serial below 2048 edges (`steps::kSerialEdges`).

**Open issues / notes**
- On LiveJournal, CPU level 2 (1755 edges, serial) took 0.28 ms in one trace, vs a few µs for comparable levels. Probably a one-off (first touch of thread-local buffers or a cold cache); not investigated.
- Swap was ~7 GB used, with other apps open; close apps before Phase 10.

### 2026-10-19 · Phase 8: PageRank

**Built**
- `src/cpu/pagerank_cpu.{hpp,cpp}`: `cpu_serial` and `cpu_omp` (pull, double, dynamic,256 schedule), with `--iters` and `--tol` modes. Also `checkRanks` (the PLAN.md 7 criteria) and `outDegrees`.
- `src/shaders/pagerank.metal` + `src/metal/pagerank_gpu.{hpp,mm}` (`gpu::GpuPageRank`). Each iteration is three steps:
  1. `pr_contrib` plus a two-pass dangling-sum reduction
  2. the main kernel
  3. `pr_diff` plus a reduction, in tol mode only

  The main kernels:
  - **`gpu_push`:** thread per source; `atomic_float` adds into `acc`; finalize computes r_new and re-zeroes `acc`.
  - **`gpu_pull_scalar`:** thread per vertex over in-edges.
  - **`gpu_pull_vector`:** simdgroup per vertex with `simd_sum`.
  - **`gpu_pull_adaptive`:** vertices binned once by in-degree (untimed preprocessing; small < 16 thread / medium simdgroup / large ≥ 256 threadgroup).
- **Fixed mode encodes init + all 20 iterations into one command buffer** (no host sync). Tol mode is one command buffer and sync per iteration.
- gbench `--algo pr`:
  - in-CSR is the transpose for directed inputs, and the stored graph itself for undirected ones (no copy)
  - prints ms/iteration and the PLAN's effective-bandwidth estimate
  - `--dump-ranks`
  - prints the final L1 change in tol mode
- `scripts/bench_pr.sh` → `results/pr.csv` (20 iterations, 2+5 runs) and `results/pr_converge.csv` (tol 1e-6, 1+3 runs).
- `tools/validate_networkx.py` now checks PageRank too (see below).
- **Tests: 22/22.** Two new tests:
  - **CPU reference vs hand-derived fixed points:** 2-vertex graph with a dangling vertex (r₀ = 0.5/1.425), a 3-cycle, a complete graph and an empty graph; error < 1e-12.
  - **Every GPU variant vs `cpu_serial`:** every synthetic graph × fixed/tol mode × tg 32/256/1024, including 4 forced adaptive threshold sets.
- **Mutation checks (each caught):** dropping the dangling term; push finalize not re-zeroing `acc`.

**Correctness problems found and fixed** (all on star100k, the adversarial case: one hub holds 46% of the rank and receives 100K identical terms)
1. **Float atomics accumulate rounding systematically.** `gpu_push` gave L1 error 1.0e-3 and ranks summing to 1.00047: 100K adds of ~1e-5 into an accumulator near 0.46 (ulp 3e-8), all rounding the same way. Real graphs were fine (L1 ~1e-7, random rounding). **Fix:** simdgroup-aggregated atomics. In each round, if all active lanes target the same vertex, they `simd_sum` and issue one atomic. Star error is now 1.9e-6. **This also made push faster:** Orkut 395 → 235 ms/iteration, LiveJournal 40.6 → 34.4, web-Google unchanged.
2. **Sequential float loops have the same problem.** `gpu_pull_scalar` gave L1 1.0e-3 on the star, because one thread sums the hub's 100K in-edges. **Fix:** Kahan summation in every per-thread in-edge loop (scalar, vector lanes, adaptive bins, and the threadgroup accumulator if a hub lands in the small bin). This is valid only because the shaders are compiled with fast math off. Cost 0–8% (LiveJournal vector 16.9 → 18.2 ms on one measurement; web-Google +3–5%; scalar unchanged).
3. **The float convergence floor can sit above tol.** On the star, the float iterate settles into a rounding cycle with L1 change 1.26e-6 (`gpu_pull_adaptive`) or 4.4e-7 (`gpu_pull_vector`), while double reaches 1e-11. Adaptive ran to the 1000-iteration cap with correct ranks (1.9e-7). **Fix:** a stagnation guard in tol mode. Stop when the L1 change hasn't improved for 5 iterations, and report "stopped at float precision floor" (`floorStop`). Accuracy is still validated against the PLAN tolerances, which are unchanged. The real graphs never hit the floor: every GPU variant converged in the same iteration count as the double reference on all six datasets.

**Test bound decision (not a PLAN tolerance):** in the GPU-vs-reference test, the converged iteration count may differ from the double reference by up to 3. Floor-stopped runs and `gpu_push` on the star are held to the rank check only.
- path1000: the per-vertex change near convergence is ~1e-9, a difference of floats near 1e-3 (ulp 1.2e-10). The float L1 delta therefore has ~10% noise, which moves the stop by 1–2 iterations (49 vs 47 observed).
- star100k: `gpu_push`'s atomic order makes its count vary (66–77 vs 90) while its ranks are accurate to 6e-7.

**NetworkX cross-check** (`results/validate_networkx.csv`): `cpu_serial` (tol 1e-10) vs `networkx.pagerank(alpha=0.85)` on all 8 synthetic graphs + ego-Facebook + web-Google. Worst L1 is 3.2e-15 and worst top-1000 relative error 8.7e-15 (web-Google), so **45/45 checks pass** (35 BFS + 10 PageRank).
- **Deviation from PLAN's `tol=1e-10` for NetworkX:** NetworkX's tol is per node (it stops at L1 < N·tol), which would allow ~1e-4 L1 error on web-Google. I used tol = 1e-10/N, i.e. L1 < 1e-10.
- 1e-12 is unreachable: on star100k, double power iteration ends in a rounding 2-cycle with L1 change 1.35e-11.

**Results: fixed 20 iterations** (`results/pr.csv`; median ms per iteration, end-to-end; all validated). **Power: battery, Low Power Mode off**; the charger was unplugged during this run. Earlier checks found GPU timings unaffected on battery but CPU ~17% slower, so treat the CPU columns as pessimistic. Rerun on AC in Phase 10. Peak RSS 2.08 GB.

| dataset | cpu_serial | cpu_omp | gpu_push | pull_scalar | pull_vector | pull_adaptive | best GPU vs cpu_omp |
|---|---|---|---|---|---|---|---|
| ego-Facebook | 0.116 | 0.102 | 0.559 | 0.133 | **0.022** | 0.059 | 4.6× |
| web-Google | 9.89 | 2.92 | 2.00 | 2.46 | **1.48** | 1.58 | 2.0× |
| roadNet-CA | 12.7 | 3.68 | 1.41 | **0.83** | 2.23 | 0.91 | 4.4× |
| soc-Pokec | 32.6 | 9.84 | 19.5 | 10.4 | 7.71 | **7.57** | 1.3× |
| soc-LiveJournal1 | 148 | 47.3 | 30.4 | 18.2 | **15.7** | 17.0 | 3.0× |
| com-Orkut | 311 | 169 | 263 | 78.0 | 57.9 | **54.1** | 3.1× |

**Convergence (tol 1e-6)** (`results/pr_converge.csv`): iterations are **identical for every variant and the double reference** on all six datasets: ego-Facebook 48, web-Google 62, roadNet-CA 57 (adaptive 58 in the first run), Pokec 44, LiveJournal 49, Orkut 45. Tol mode adds the diff kernel and a ~0.1–0.25 ms per-iteration sync. Best GPU convergence times: LiveJournal 0.87 s (`gpu_pull_vector`) vs 2.44 s (`cpu_omp`); Orkut 2.24 s vs 6.45 s.

**Effective bandwidth** (PLAN formula: bytes = 4(n+1) + 8m + 4n per iteration, over e2e time; best pull variant per graph). "Uniform-random rate" is the Phase 3 GPU random-gather rate for a table the size of `contrib`.

| dataset | variant | GB/s | % of 95 GB/s sequential | gathers/s | contrib size | Phase 3 uniform-random rate | ratio |
|---|---|---|---|---|---|---|---|
| roadNet-CA | scalar | 72.1 | **76%** | 6.65 G | 7.9 MB | ~1.6 G | 4.2× |
| web-Google | vector | 32.4 | 34% | 3.45 G | 3.5 MB | ~2.7 G | 1.3× |
| soc-Pokec | adaptive | 34.1 | 36% | 4.04 G | 6.5 MB | ~1.6 G | 2.5× |
| soc-LiveJournal1 | vector | 37.4 | 39% | 4.36 G | 19 MB | ~1.1–1.6 G | 2.7–4.0× |
| com-Orkut | adaptive | 35.1 | 37% | 4.33 G | 12 MB | ~1.1 G | 3.9× |

- **Social graphs reach ~35–39% of sequential bandwidth.** Half of the counted bytes (the `contrib` gathers) are random 4-byte reads, each pulling a cache line.
- **Their gather rate is 2.5–4× the uniform-random rate for the same table size.** Real orderings have locality: in-neighbors' IDs cluster, so many gathers hit lines already in cache. Phase 9 tests how much of that comes from the ordering.
- **roadNet-CA reaches 76% of sequential**, the highest of any graph. Its SNAP ordering is strongly local (neighbors have nearby IDs), so gathers behave almost like streaming.

**Conclusions**
- **Push vs pull: pull wins on every graph with skewed in-degree.** `gpu_push` is 1.9× slower than the best pull on LiveJournal, 2.5× on Pokec and 4.9× on Orkut. Each edge is a random atomic read-modify-write (2× the memory traffic of a gather, plus contention on hubs), while pull does one plain gather per edge plus one store per vertex. Push only comes close on web-Google (2.00 vs 1.48) and roadNet-CA (1.41 vs 0.83): low degrees, little contention. Push also needed simdgroup aggregation to be *accurate*, and it is the only non-deterministic variant.
- **Scalar vs vector: it depends on degree, as expected.** Vector (32 lanes per vertex, coalesced `col_idx`) beats scalar by 16–34% on the social graphs and 40% on web-Google, but is **2.7× slower on roadNet-CA**. There the average in-degree is 2.8, so ~90% of each simdgroup's lanes idle. Scalar is the reverse.
- **Adaptive is the robust choice: within 10% of the best variant on every non-tiny graph, and the best on Pokec and Orkut.** It keeps roadNet-CA near scalar (0.91 vs 0.83) and the social graphs near or better than vector. My first adaptive used the BFS-style cooperative small bin and was 1.7× slower than scalar on roadNet (1.43 ms); replacing it with a plain thread-per-vertex loop fixed that, with no loss elsewhere (measured). On ego-Facebook (4K vertices) three dispatches cost more than one.
- **GPU vs CPU:** the best GPU variant is 2.0–4.6× faster than 8-thread `cpu_omp`, except Pokec (1.3×). CPU numbers were measured on battery.

**Open issues / notes**
- The Orkut `gpu_pull_vector` fixed-mode run (57.9 ms/iteration) was slower than its own tol-mode run (49.7 ms/iteration, which includes the diff kernel). I suspect a sustained-load or thermal effect on battery (the fixed run is one 1.2 s command buffer). Recheck on AC in Phase 10.
- All Phase 8 timings were measured on battery (Low Power Mode off). Rerun on AC in Phase 10.

### 2026-10-19 · Phase 9: vertex reordering

**Built**
- `src/graph/reorder.{hpp,cpp}` (`graph::reorder`): `compute(sym, ordering)` returns perm[old] = new for each ordering:
  - `original`: identity
  - `random`: Fisher–Yates, fixed seed
  - `degree`: descending degree, stable
  - `rcm`: Reverse Cuthill–McKee. Per component: start at the lowest-degree unvisited vertex, refine once to the lowest-degree vertex of the farthest BFS level (GPS-style pseudo-peripheral start), visit neighbors in ascending degree, then reverse the whole order.

  Also `isPermutation` and `locality()` (mean |u−v| over adjacency entries, and the fraction of entries with u/64 == v/64).
- `graph::permute(g, perm)` in `transform`: relabels rows and columns and re-sorts lists (parallel over rows).
- **gbench `--ordering`** for BFS and PageRank, plus `--locality-csv`.
  - Orderings are computed on the symmetrized graph; PageRank permutes the directed graph with the same perm.
  - **Validation in original IDs:** the reference runs on the *original* graph with the original source, and the result on the relabeled graph is mapped back (`dist_orig[v] = dist_run[perm[v]]`, same for ranks). Sources and TEPS denominators are identical across orderings.
  - Memory (Orkut): the PageRank reference is computed before building the relabeled graph, and the relabeled host copy is dropped after the GPU upload. Peak RSS for the whole sweep is 2.69 GB, under the 3 GB limit.
- `scripts/bench_reorder.sh` → `results/reorder_bfs.csv`, `results/reorder_pr.csv`, `results/reorder_locality.csv`, `results/trace_reorder_access.csv`. Extra traces: `results/trace_reorder_hybrid.csv`.
- **Tests: 24/24.** New:
  - `reorder_valid_and_results_map_back`: every ordering on every synthetic graph is a permutation; permute + inverse restores the graph exactly; BFS distances map back exactly; PageRank ranks map back to rounding (1e-12 relative).
  - `reorder_locality_known_answers`: RCM recovers a randomly labeled path exactly (mean gap 1.0); degree order puts the star's hub at id 0; random labels raise a grid's mean gap >10×.

**Validation:** all 76 runs (6 datasets × 4 orderings × {hybrid BFS, `gpu_pull_vector`, `gpu_pull_adaptive`}, plus the access-order runs) pass after mapping back to original IDs. AC power, Low Power Mode off.

**Locality metrics** (`results/reorder_locality.csv`, symmetrized graph; mean |u−v| / % of entries within one 64-vertex block)

| dataset | original | random | degree | rcm |
|---|---|---|---|---|
| ego-Facebook | 289 / 9.3% | 1344 / 1.5% | 546 / 7.1% | 210 / **19.8%** |
| web-Google | 296K / 0.008% | 292K / 0.008% | 204K / 0.13% | 87K / **14.2%** |
| roadNet-CA | 4.1K / **61.5%** | 655K / 0.003% | 416K / 33.0% | **3.0K** / 12.6% |
| soc-Pokec | 305K / 0.79% | 544K / 0.004% | 292K / 0.06% | 322K / **1.68%** |
| soc-LiveJournal1 | 754K / **7.1%** | 1.62M / 0.001% | 660K / 0.61% | 774K / 5.9% |
| com-Orkut | 632K / 1.94% | 1.02M / 0.002% | 573K / 0.08% | 643K / 1.92% |

**web-Google's SNAP labels are effectively random** (same locality as `random`), so original ≈ random there. roadNet-CA's original order is already highly local, and its same-block fraction even beats RCM's. RCM's computation time ranges from 1 ms (ego-Facebook) to 1.6 s (Orkut), untimed.

**Results** (median: BFS e2e ms, 64 sources × 5 trials; PageRank ms/iteration, 20 iterations × 5 trials)

BFS (`hybrid`, T = 1e5):

| dataset | original | random | degree | rcm | best vs original | best vs random |
|---|---|---|---|---|---|---|
| ego-Facebook | **0.85** | 0.86 | 0.99 | 1.07 | original 1.00× | 1.02× |
| web-Google | 6.69 | 6.35 | 4.94 | **4.69** | rcm 1.43× | 1.36× |
| roadNet-CA | 22.2 | 38.3 | 22.5 | **19.8** | rcm 1.12× | 1.93× |
| soc-Pokec | 4.67 | 5.73 | **4.60** | 5.80 | degree 1.02× | 1.25× |
| soc-LiveJournal1 | **8.64** | 15.9 | 9.57 | 12.2 | original 1.00× | 1.84× |
| com-Orkut | **12.8** | 17.4 | 13.1 | 20.3 | original 1.00× | 1.37× |

PageRank `gpu_pull_vector`:

| dataset | original | random | degree | rcm | best vs original | best vs random |
|---|---|---|---|---|---|---|
| ego-Facebook | **0.044** | 0.070 | 0.048 | 0.051 | original 1.00× | 1.59× |
| web-Google | 1.43 | 1.48 | 1.25 | **1.10** | rcm 1.30× | 1.35× |
| roadNet-CA | **2.23** | 2.69 | 2.33 | 2.29 | original 1.00× | 1.21× |
| soc-Pokec | 6.89 | 11.1 | 6.94 | **5.76** | rcm 1.20× | 1.93× |
| soc-LiveJournal1 | 14.9 | 38.2 | 20.6 | **13.6** | rcm 1.10× | 2.80× |
| com-Orkut | 49.2 | 107.0 | 63.6 | **44.2** | rcm 1.11× | 2.42× |

PageRank `gpu_pull_adaptive`:

| dataset | original | random | degree | rcm | best vs original | best vs random |
|---|---|---|---|---|---|---|
| ego-Facebook | 0.059 | 0.080 | 0.065 | **0.057** | rcm 1.05× | 1.41× |
| web-Google | 1.56 | 1.60 | 0.94 | **0.67** | rcm **2.32×** | 2.37× |
| roadNet-CA | 0.894 | 2.70 | 1.11 | **0.890** | rcm 1.00× | **3.03×** |
| soc-Pokec | 7.37 | 11.7 | 7.06 | **6.03** | rcm 1.22× | 1.94× |
| soc-LiveJournal1 | 15.3 | 40.2 | 20.2 | **14.5** | rcm 1.06× | 2.78× |
| com-Orkut | 50.3 | 102.7 | 63.4 | **44.6** | rcm 1.13× | 2.30× |

**Interpretation**
- **The input orderings already carry most of the locality.** Random labels slow PageRank 2.3–2.8× on LiveJournal and Orkut and 3.0× on roadNet-CA (adaptive), and BFS 1.4–1.9×. Gathers drop from ~4.6 to ~1.8 G/s on LiveJournal, which is close to the Phase 3 uniform-random rate for a 19 MB table (1.1–1.6 G/s). That confirms Phase 8's point: the 2.5–4× above-random gather rate came from the ordering.
- **RCM is the best ordering for PageRank:** 1.06–1.22× on the social graphs and up to **2.32×** on web-Google, whose original labels are random. There, `gpu_pull_adaptive` reaches 71 GB/s effective (75% of sequential), vs 31 GB/s with original labels. On roadNet-CA, RCM merely ties the already-local original order.
- **Degree ordering hurts PageRank on the community-structured social graphs** (LiveJournal 0.72–0.76×, Orkut 0.77–0.79×). Grouping hubs breaks up community locality (LiveJournal same-block 7.1% → 0.6%). It helps only web-Google (1.14–1.66×), which had no locality to lose.
- **BFS behaves differently: RCM hurts direction-optimizing BFS on social graphs** (LiveJournal 0.71×, Orkut 0.63×, Pokec 0.81×) while helping web-Google (1.43×) and roadNet-CA (1.12×). The cause (verified with traces, `results/trace_reorder_hybrid.csv`, LiveJournal, 8 sources, medians):

  | ordering | bottom-up edges examined | bottom-up ms | top-down edges | top-down ms |
  |---|---|---|---|---|
  | original | 22.2M | 7.8 | 1.46M | 2.1 |
  | degree | **21.8M** | **7.3** | 1.46M | 2.4 |
  | random | 28.5M | 11.6 | 1.46M | 3.4 |
  | rcm | **44.3M** | 10.1 | 1.46M | 1.9 |

  Bottom-up stops at the first neighbor in the frontier, and adjacency lists are sorted by ID. So **neighbor order decides how early the scan hits.** Original social labels and degree order put early and high-degree vertices at low IDs; they join the frontier early, so scans hit quickly. RCM gives the lowest IDs to the peripheral vertices visited last, and **bottom-up examines 2× more edges**. Top-down work is ordering-independent (it examines every frontier edge), and RCM even makes it the fastest (1.9 ms) through locality. For BFS, an ordering's effect on early exit can outweigh its locality effect. A hub-first neighbor order inside each list, independent of the vertex labels, would get both; noted for later, not done.
- **No ordering is best for both algorithms:** PageRank wants RCM, direction-optimizing BFS wants hub-first labels (original/degree). On web-Google and roadNet-CA, RCM wins both.

**Phase 5 open question (why `gpu_vertex` beat queue variants on late levels): mostly answered** (`results/trace_reorder_access.csv`, LiveJournal, 4 sources, per-level medians in ms)

| level | gpu_edge original | gpu_vertex original | gpu_edge random | gpu_vertex random |
|---|---|---|---|---|
| 5 | 16.7 | **11.1** | 30.3 | 27.1 |
| 6 | 12.7 | **5.1** | 24.4 | 13.0 |
| total | 38.3 | 31.1 | 72.7 | 69.4 |

- **Level 5:** `gpu_vertex`'s 1.5× advantage almost disappears under random labels (1.12×). So it came mostly from the input ordering's locality, which ID-order traversal exploits.
- **Level 6:** the ~1.9× advantage survives random labels. That is consistent with an effect no relabeling can remove: `gpu_vertex` always streams `col_idx` in storage order (thread v reads row v), while queue variants read rows in discovery order. Plausible, not proven.
- Overall, `gpu_vertex`'s lead shrinks from 7.2 ms to 3.3 ms under random labels.

**Decisions**
- Orderings are computed on the symmetrized graph, and the same perm is used for PageRank's directed graph.
- Relabeling and RCM are preprocessing, excluded from timing (RCM up to 1.6 s on Orkut, recorded in the locality CSV).
- Validation always uses original-label references; results are mapped back.

**Open issues / notes**
- Hub-first ordering *within* adjacency lists (keeping vertex labels) could combine RCM's PageRank locality with fast bottom-up exits. Not done; outside the PLAN.
- Swap was ~6 GB used with other apps open; Phase 10 should run with apps closed.

### 2026-10-19 · Phase 10: full benchmark run, figures, report

**Built**
- `scripts/sweep_tg.sh`: threadgroup sizes 64–1024 for `gpu_do` (LiveJournal, Pokec, Orkut) and `gpu_pull_vector`/`gpu_pull_adaptive` (four graphs) → `results/sweep_tg.csv`.
- `scripts/run_all.sh`: the full validated matrix in stages (test, micro, stats, bfs_cpu, scaling, bfs_gpu, star, hybrid_sweep, pr, reorder, networkx, plots; tuning sweeps only when named). It prints an estimate, refuses to run on battery or Low Power Mode unless `FORCE=1`, and logs to `logs/` (gitignored). The GPU BFS variants now share `results/bfs_gpu.csv` and `results/trace_bfs.csv`; the per-phase `bfs_gpu_do.csv`, `bfs_hybrid.csv` and `trace_bfs_*` files were retired (still in git history).
- `analysis/plots.py`: figures 1–11 as PNG + SVG. `analysis/report_tables.py` generates the markdown tables in the report.
- `docs/REPORT.md` and the final `README.md`.

**Threadgroup sweep → defaults unchanged (256).** All differences are ≤ 5% for PageRank and ≤ 10% for `gpu_do`, with no consistent winner (e.g. Orkut `gpu_do`: 17.8 / 17.3 / 17.2 / 16.7 / 18.5 ms for tg 64…1024). The kernels are latency-bound, not occupancy-bound.

**Intermittent `gpu_push` failure (fixed in 7a74ade).** The first `run_all` aborted in the test stage. `gpu_push` in tol mode on star100k gave L1 1.02e-4 against the 1e-4 limit, in 1 of 15 stress runs. The simdgroup-aggregated atomics from Phase 8 remove most systematic rounding, but rounds where lanes target different vertices still add one at a time into the hub's accumulator, and the error depends on atomic order. **Fix:** compensated atomics. After each `atomic_fetch_add`, the rounding error of that add is computed exactly (TwoSum on the returned old value) and, if it is larger than 2^-16 of the term, added to a second accumulator `acc_lo`; finalize uses `acc + acc_lo`. Compensating every add cost 2.2×; the threshold makes it 0–12%. Star L1 is now 4e-7, 0/15 failures under stress, 4/4 clean full-suite repeats. No tolerance was changed.

**Full run** (`logs/run_all_20261019_173954.log`; AC power, Low Power Mode off, other apps closed; all stages passed, 73 min; `make test` 24/24; NetworkX 45/45; peak RSS 2.7 GB). Every number in the report comes from this run. The Phase 6 and 8 tables, which were measured on battery, are superseded. Main changes vs the earlier runs:
- `cpu_do` on AC: LiveJournal 20.7 ms, Orkut 25.1 (Phase 2: 22.4 / 28.0). `gpu_do` 10.7 / 13.3; hybrid 8.76 / 13.1; Pokec hybrid 4.76.
- PageRank on AC: `cpu_omp` LiveJournal 42.5 ms/iteration (battery 47.3). `gpu_pull_vector` is now the best variant on Pokec and Orkut too (6.84, 49.0), with adaptive within 10%.
- **Orkut vector anomaly (Phase 8) resolved:** fixed mode 49.0 vs tol mode 49.9 ms/iteration on AC. It was a battery effect.
- web-Google `hybrid` 6.42 ms is now slightly ahead of `cpu_do` (6.95).

**Report findings worth remembering**
- The hypothesis "small graphs have higher GTEPS (cache)" is **wrong for BFS**. GTEPS grows with size, because small graphs are bound by per-level fixed costs (fig 10).
- roadNet-CA: `hybrid` (22.0 ms) is effectively a 4-thread CPU BFS. It matches `cpu_omp` at 4 threads (20.1 ms). Eight threads are slower (35–39 ms) for every schedule because of E-core barriers (fig 7).
- Static scheduling is 24% slower than dynamic on LiveJournal at 8 threads (18% already at 2 threads).
- Best BFS efficiency: 84 MTEPS per GB/s (Orkut, hybrid); PageRank reaches up to 78% of the measured 95 GB/s (roadNet-CA scalar).

**Figure fix:** fig 10 originally joined points in dataset order, making zig-zag lines, and its labels fell outside the axes. Points are now sorted by size, labels are inside the axes, and the legend is outside the plot.

**Open issues / notes**
- ego-Facebook PageRank at ~0.02–0.06 ms/iteration varies up to 2× between runs (`pr.csv` vs `reorder_pr.csv`). These numbers are noted as noise in the report's limitations; no conclusions are drawn from them.
- Not done (future work in the report): indirect dispatch to cut per-level syncs, hub-first neighbor order within lists, P-core-only OpenMP, Phase 11 GAP baseline.

### 2026-10-19 · Phase 11: external reference (GAP Benchmark Suite)

**Built**
- `scripts/gap.sh`:
  - clones gapbs into `external/gapbs` (gitignored) at a pinned commit (941462c)
  - builds `bfs` and `pr` with Apple clang + libomp by overriding `CXX_FLAGS` on the make command line (`-Xpreprocessor -fopenmp`, libomp include/lib, `-O3 -march=native -mmacosx-version-min=26.0`). No GAP source or Makefile is patched, and the build has 0 warnings.
  - checks power, then runs `tools/run_gap.py`
- `build/gexport` (`src/graph/export_sg.cpp`): `.bin` → GAP `.sg` (bool directed, int64 m, int64 n, int64 offsets, int32 neighbors, plus in-CSR if directed). `--sym` writes the symmetrized graph for BFS. GAP therefore runs on exactly our graphs (web-Google: GAP reports 4,322,051 undirected edges, same as our stats).
- `tools/run_gap.py`: exports each graph to `build/sg/` (deleted after use, so `data/` never grows). Then:
  - **BFS:** runs GAP from **our 64 sources** (`-r`, `-n 6`, first trial dropped as warm-up) at 8 and 4 threads, with GTEPS from our per-source component-edge denominators.
  - **PageRank:** fixed 20 iterations (`-i 20 -t 0`, 2 warm-up + 5 trials) at 8 and 4 threads, plus time to tol 1e-6 (`-l` counts iterations).
  - **Correctness:** GAP's verifier, one `-v` run per BFS source and per PageRank graph. All passed.
  - Output: `results/gap_bfs.csv`, `gap_pr.csv`, `gap_pr_converge.csv` in our schema (variants `gap_bfs`, `gap_bfs_4t`, `gap_pr`, `gap_pr_4t`).
- `plots.py` / `report_tables.py` include `gap_bfs` / `gap_pr` when the CSVs exist (figs 4 and 8; new `converge` table). The efficiency table now picks the best variant from all variants, including GAP.
- Run: ~6 min, peak RSS 1.0 GB, AC power.

**Results**

| | Facebook | web-Google | roadNet-CA | Pokec | LiveJournal | Orkut |
|---|---|---|---|---|---|---|
| BFS cpu_do (8t), ms | 0.36 | 6.95 | 38.0 | 8.07 | 20.7 | 25.1 |
| BFS GAP (8t) | 0.35 | 6.23 | 37.3 | 8.04 | 20.0 | 24.7 |
| BFS GAP (4t) | 0.13 | 7.44 | **21.1** | 9.09 | 25.3 | 27.9 |
| BFS hybrid | 0.84 | 6.42 | 22.0 | **4.76** | **8.76** | **13.1** |
| PR cpu_omp ms/it | 0.10 | 2.67 | 3.11 | 10.2 | 42.5 | 138 |
| PR GAP (8t) ms/it | 0.13 | 1.66 | 2.06 | 6.84 | 21.1 | 51.1 |
| PR best GPU ms/it | 0.021 | 1.43 | 0.80 | 6.84 | 14.0 | 49.0 |
| PR to 1e-6, GAP ms (iters) | 4.50 (33) | 53.8 (32) | 65.0 (34) | 197 (29) | 650 (31) | 1522 (29) |
| PR to 1e-6, best GPU ms (iters) | 9.51 (48) | 107 (62) | 72.7 (57) | 320 (44) | 719 (49) | 2247 (45) |

**Findings**
- **Our `cpu_do` is a faithful GAP-class baseline.** It is within 0–2% of GAP's BFS everywhere except web-Google (GAP 10% faster). The hybrid beats GAP 1.7–2.3× on the social graphs.
- **roadNet-CA:** GAP at 4 threads (21.1) edges out the hybrid (22.0), and GAP at 8 threads is 1.8× slower than at 4. This confirms the Phase 10 reading: the hybrid is a CPU BFS there, and E-cores hurt.
- **Our `cpu_omp` PageRank was a weak baseline:** GAP is 1.5–2.7× faster per iteration. Causes visible in the code: GAP gathers float contributions (half the footprint), computes contributions in the same loop that updates scores, and ours does a serial copy of the rank vector each iteration.
- **Against GAP, the GPU's per-iteration PageRank lead is 1.0–2.6×** (ties on Pokec and Orkut).
- **To convergence, GAP wins on every graph (1.1–2.1×).** In-place Gauss–Seidel needs 29–34 iterations vs 44–62 for Jacobi.
- **Caveats** (in the report): GAP's PageRank has no dangling redistribution, so it converges to a different vector. Its in-sweep stopping test makes iteration counts nondeterministic (Orkut 23–30).
- **Report changes:**
  - updated the abstract, 4.3, 4.7, efficiency and limitations
  - added section 4.12
  - moved "run GAP" out of future work and added "Gauss–Seidel PageRank on the GPU"
  - the "1.5–4.8× faster than CPU PageRank" claim is now qualified as being against our `cpu_omp`

**Open issues / notes**
- None blocking. Possible follow-ups: GPU in-place (Gauss–Seidel) PageRank; P-core-only OpenMP; indirect dispatch for BFS.

### 2026-10-19 · Phase 12 (extension): in-place (Gauss–Seidel-style) GPU PageRank

Not in PLAN.md. Requested after Phase 11 showed GAP's Gauss–Seidel PageRank converging first on every graph.

**Built**
- **Shader:** pull kernels take `PrParams::inplace`. `pr_store` writes `r_new[v]` and, in place mode, also `contrib[v] = r/outdeg`, so later threads in the same sweep pull new values. New kernels `pr_sum` / `pr_scale` renormalize the ranks to sum 1 after each sweep (in-place mode only). The Jacobi kernels are unchanged in speed (A/B on the same binary, LiveJournal: vector 14.1 vs 14.2, adaptive 15.2 vs 15.7 ms/iteration).
- **`gpu_pull_gs`** (`GpuPageRank::prepareGS`) picks an update order per graph:
  - **colored:** ≤ 8 Jones–Plassmann rounds on the in-CSR (hash priorities), used if they cover ≥ 95% of vertices. Runs on a private relabeled copy of the in-CSR so that each set is a contiguous ID range; `ranks()` maps the result back lazily (untimed).
  - **natural:** otherwise, one in-place `gpu_pull_vector` dispatch.
- **gbench additions:**
  - `--inplace`: any pull variant updates in place (experiments)
  - `--ref-tol`: the tolerance of the Jacobi reference used to validate GS runs
  - tol in `params` is now printed with `%g`. It used to be `to_string(tol).substr(0,8)`, so every tol < 1e-6 was recorded as `0.000000`.
- `scripts/bench_gs.sh` → `results/pr_gs.csv` (fixed 20), `pr_gs_equal.csv` (equal-accuracy runs), `pr_gs_ablation.csv` (in-place with natural vs degree-bin order). `run_all.sh gs` runs it (optional stage).
- `report_tables.py gs gs_ablation`; report section 4.13.
- **Tests: 25/25.** New `pagerank_gs_converges_to_reference`: every synthetic graph, both orders, tg 32/256/1024. It checks:
  - 20 sweeps: sum = 1
  - 200 sweeps: PLAN tolerances vs the converged reference
  - tol 2.5e-7: no less accurate than Jacobi at 1e-6

  Mutation checks (each caught): no renormalization; a wrong in-place contribution.

**Validation rule for GS (no PLAN tolerance was changed)**
- PLAN.md 7 compares a variant with `cpu_serial` *run the same way*, i.e. the same Jacobi trajectory, so in practice it measures float rounding. A Gauss–Seidel iterate is a different vector, so gbench compares it with the **converged** reference (tol 1e-10) instead:
  - **tol mode:** `cpu::gsNoWorseThanJacobi`. Each metric must be within its PLAN tolerance or no worse than the Jacobi reference's own distance from convergence. The sum gets the L1 budget, since |sum−1| ≤ L1 and Jacobi conserves mass exactly. The top-1000 relative bound gets 1e-4 relative slack for float-vs-double rounding (roadNet after 20 iterations: both 0.022463 at the same unconverged vertex).
  - **fixed mode:** sum = 1 per trial, plus an untimed 200-sweep run that must meet the PLAN tolerances vs converged.
- **Why not "20 GS sweeps no worse than 20 Jacobi"?** It is false on ego-Facebook (1.1× worse): that is a performance property, not correctness.
- **Why 200 sweeps, not a tol run?** See the 2-cycle finding below.

**Findings, in the order they came up**
1. **First version** (in place over the adaptive degree bins, no renormalization): slower than Jacobi on ego-Facebook (56 vs 48 sweeps). A numpy model of block orders reproduced it, and suggested that the degree-bin order was the problem and that ID-ordered chunks or coloring would help.
2. **Coloring** (numpy): roadNet-CA 57 → 29 sweeps with 8 independent sets. It is useless on the social graphs: after 16 rounds, 67–86% of vertices remain uncolored.
3. **Interleaved color sets thrash the cache:** roadNet-CA took 4.2 ms per sweep (vs 1.3) until the sets were relabeled to contiguous ranges (1.4–1.6 ms).
4. **Mass drift:** on rmat12 the GS error after 20 sweeps was 6e-4 vs Jacobi's 6e-8. The total rank drifts with in-place updates, and the drift decays only by d per sweep. Renormalizing after each sweep fixed it (9e-7). **This also invalidated finding 1:** with renormalization, the degree-bin order needs about the same sweeps as natural order (ablation table). The numpy model had the same flaw. The code comment was corrected.
5. **Uneven convergence (the main result).** soc-Pokec at tol 1e-7 met L1 easily but had top-1000 relative error 2.9e-3. The cause is the 2-cycle 151826 ↔ 151828 (each has outdeg 1, pointing at the other). Being in the same wave, the pair decays at the Jacobi rate d while the rest of the graph converges faster, so the global L1 test stops ~20 sweeps earlier and leaves the pair ~30× off (0.85^-21). The same effect makes GS need a much tighter tol to match Jacobi's accuracy.
6. **Equal-accuracy benchmark** (strict: L1 and top-1000 relative error both ≤ Jacobi@1e-6). Two full runs gave the same verdict per graph, with speedups within 5%:

| dataset | order | GS tol needed | GS ms (sweeps) | best Jacobi ms (it) | speedup | GAP ms |
|---|---|---|---|---|---|---|
| Facebook | natural | 5e-7 | 11.7 (52) | 9.51 (48) | 0.81× | 4.50 |
| web-Google | natural | 3.1e-8 | 88.6 (46) | 107 (62) | 1.21× | 53.8 |
| roadNet-CA | colored | 2.5e-7 | 63.6 (33) | 72.7 (57) | 1.14× | 65.0 |
| Pokec | natural | 3.1e-8 | 352 (46) | 320 (44) | 0.91× | 197 |
| LiveJournal | natural | 1.25e-7 | 901 (52) | 719 (49) | 0.80× | 650 |
| Orkut | natural | 1e-6 | 1618 (32) | 2247 (45) | 1.39× | 1522 |

- **Same-tol sweeps** (1e-6): web-Google 62 → 33, Pokec 44 → 25, Orkut 45 → 32, roadNet 57 → 29, but GS is less accurate at those stops.
- **After 20 sweeps, GS is closer to converged than Jacobi:** 28× Pokec, 24× web-Google, 19× roadNet-CA, 4× LiveJournal/Orkut; 1.1× worse on Facebook.
- **Verdict:** a partial win. It does not close the gap to GAP except on roadNet-CA (tie). The report's future work now lists orders that separate mutually dependent vertices, and a per-vertex or sampled-residual stopping test.

**Other notes**
- The first `make test` attempts looked hung: the test process had been started as a zsh background job, which zsh runs at low priority (BG_NICE), while system Contacts daemons used ~90% of a core. The hybrid BFS test (one GPU sync per level) crawled. In the foreground the suite takes 1:39. Benchmarks use the harness's background mode, which is not niced (verified: gbench at nice 0). The final suite run (25/25) took 18 min at 2% CPU after the harness moved it to the background at its 10-min limit; the cause is not identified. GPU-sync-heavy tests crawl there. Benchmarks launched in background mode from the start produced timings consistent with foreground runs (e.g. GAP and run_all vs earlier phases).

### 2026-10-20 · Phase 13 (extension): fixing the in-place PageRank stragglers

Requested after Phase 12: try (a) an update order that separates mutually dependent vertices and (b) a per-vertex stopping test. Also built a presentation website.

**Diagnosis first** (dumped GS@1e-6 and converged ranks; looked at the 10 worst top-1000 vertices per graph): almost all were **mutual pairs with out-degree 1–2** (each passes most of its rank to the other):
- Pokec 151826↔151828 and LiveJournal 341090↔341091, both in the same 32-vertex block
- web-Google 873801 with 750372 and 875012
- Orkut (undirected, high degree): none

**Built**
- `GpuPageRank::prepareGS(pairMaxOutdeg = 2)`: in natural order, one vertex of each mutual pair with both out-degrees ≤ K goes to a second dispatch (simdgroup per vertex over a list: `pr_pull_medium`) after the main list. The choice is greedy in ID order, so the deferred set contains no pair. gbench `--gs-pairs K` (0 = off).
- Per-vertex stopping test: `pr_diff` also computes max |Δ| / max(r, 1/n) (uint max on float bits, `pr_reduce_max` → `scalars[3]`). `GpuPrOptions::tolRel` / gbench `--tol-rel R` stops only when the L1 change < tol AND max < R. `GpuPrResult::lastMaxRel`.
- `scripts/bench_gs.sh` rewritten: runs 9 candidate stopping rules per graph (L1 tols 1e-6…3.1e-8, and tol 1e-6 + tol-rel 1e-4/5e-5/2.5e-5), logs all to `results/pr_gs_search.csv`, keeps the fastest strict equal-accuracy run in `pr_gs_equal.csv`, and records K=0 vs K=2 in `pr_gs_pairs.csv`.
- Test `pagerank_gs_converges_to_reference` now loops K ∈ {0, 2}, adds the tol + tolRel rule, and checks that some graph has deferred vertices (rmat12). 25/25.
- Report section 4.14; the abstract, hypothesis table, future work and README are updated.

**Bugs found on the way**
- `run()` called `prepareGS()` with its default argument, silently overriding the caller's K, so the first "K=0" runs were really K=2 and looked impossibly good. Caught by rebuilding HEAD in a git worktree, which reproduced the old 1.46e-2 exactly. Fixed: `run()` only prepares if not already prepared.
- The first deferral version deferred both 873801 and 875012 (partners of each other) in web-Google, so they ran concurrently again; hence the greedy rule.
- `|Δ|/r` as the per-vertex measure was dominated by float noise on tiny-rank vertices; changed to `max(r, 1/n)`.
- `bench_gs.sh`: `grep` with no match under `pipefail` aborted the script on colored-order graphs (no `deferred=` field); fixed with `|| true`. Re-ran roadNet-CA onward.

**Results** (worst top-1000 rel error at tol 1e-6, K=0 → K=2): web-Google 9.0e-4 → 1.0e-4, Pokec 1.46e-2 → 7.3e-4, LiveJournal 1.81e-3 → 1.9e-5, Orkut unchanged. No extra sweeps; each sweep costs 10–17% more.

Equal accuracy vs the best Jacobi GPU kernel (Phase 12 → Phase 13):

| dataset | Phase 12 | Phase 13 | rule | GAP |
|---|---|---|---|---|
| Facebook | 0.81× | 0.78× (12.2 ms) | tol 2.5e-7 | 4.50 ms |
| web-Google | 1.21× | **1.40×** (76.4 ms) | tol 1e-6 + rel 5e-5 | 53.8 ms |
| roadNet-CA | 1.14× | 1.09× (66.5 ms; same algorithm, noise) | tol 2.5e-7 | 65.0 ms |
| Pokec | 0.91× | **1.68×** (190 ms) | tol 1e-6 | 197 ms |
| LiveJournal | 0.80× | 0.99× (730 ms) | tol 5e-7 | 650 ms |
| Orkut | 1.39× | **1.40×** (1609 ms) | tol 1e-6 + rel 5e-5 | 1522 ms |

**Per-vertex test verdict:** the fastest safe rule on web-Google and Orkut. On LiveJournal and Pokec it over-iterates (45–56 sweeps), because some low-rank vertex keeps moving ~0.2% after the L1 test fires. So it is a stronger guarantee, not a universal speed-up.

**Website:** `docs/index.html`. It has the problem statement, an explanation section with two interactive demos (a BFS top-down/bottom-up stepper, and a PageRank Jacobi / GPU-wave / deferral / sequential-GS simulator on a 1,000-vertex model graph, prototyped in Node to check it shows the effect), setup, approach timeline, all results with figures, findings, the hypothesis table, conclusion, limits and reproduction steps. Also published privately as a claude.ai artifact.
