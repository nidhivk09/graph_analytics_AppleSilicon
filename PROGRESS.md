# PROGRESS

Running log. A fresh session should be able to continue from `PLAN.md` + this file.

## Current state

- **Phase 2 complete** (2026-10-14). Waiting for approval to start Phase 3.
- Next: Phase 3 (gmicro --bandwidth / --launch / --random microbenchmarks).
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
