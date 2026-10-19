# PROGRESS

Running log. A fresh session should be able to continue from `PLAN.md` + this file.

## Current state

- **Phase 4 complete** (2026-10-15; timings rerun on AC power 2026-10-19). Waiting for approval to start Phase 5.
- Next: Phase 5 (`gpu_binned`, `gpu_edge` load-balanced BFS).
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
