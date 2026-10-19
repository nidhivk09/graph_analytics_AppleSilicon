# graph-gpu: BFS and PageRank on Apple Silicon (Metal)

Breadth-first search and PageRank as Metal compute kernels on an 8 GB Apple M2,
compared with OpenMP CPU baselines and the GAP Benchmark Suite on six SNAP graphs (4K to 234M edges). Every
GPU result is validated against a serial CPU reference: BFS must match exactly,
and PageRank must have L1 error < 1e-4. The full write-up with all figures is
**[docs/REPORT.md](docs/REPORT.md)**. `PLAN.md` holds the design and
`PROGRESS.md` the phase-by-phase log.

## Headline results (M2, 10-core GPU, measured 95 GB/s)

| | LiveJournal (69M edges) | Orkut (234M) | roadNet-CA (5.5M, 600+ levels) |
|---|---|---|---|
| BFS `cpu_serial` | 401 ms | 541 ms | 36.7 ms |
| BFS `cpu_do`, 8 threads | 20.7 ms | 25.1 ms | 38.0 ms |
| BFS GAP, 8 threads (4 threads) | 20.0 ms | 24.7 ms | 37.3 ms (**21.1 ms**) |
| BFS `gpu_do` (direction-optimizing, GPU only) | 10.7 ms | 13.3 ms | 113 ms |
| BFS `hybrid` (small levels on CPU, shared buffers) | **8.8 ms (4.6 GTEPS)** | **13.1 ms (8.0 GTEPS)** | **22.0 ms** |
| PageRank `cpu_omp`, ms/iteration | 42.5 | 138 | 3.11 |
| PageRank GAP, 8 threads, ms/iteration | 21.1 | 51.1 | 2.06 |
| PageRank best GPU pull, ms/iteration | **14.0** | **49.0** | **0.80** |
| PageRank to tol 1e-6: GAP vs best GPU | **650** vs 719 ms | **1522** vs 2247 ms | **65** vs 73 ms |

- **Direction optimization is the big win on social graphs.** `gpu_do` is 3–5×
  faster than load-balanced top-down GPU BFS. The hybrid is 1.7–2.3× faster than
  GAP's 8-thread BFS (which our `cpu_do` matches) and 41–46× faster than serial.
- **High-diameter graphs are bound by per-level sync, not bandwidth.** Each GPU
  level costs ~0.2 ms however small it is. On roadNet-CA the hybrid runs those
  levels on the CPU at zero copy cost and beats GPU-only BFS by 5.1×, though
  GAP on the 4 P-cores alone is as fast.
- **PageRank: pull beats atomic push by 1.5–4.9×.** Simdgroup-per-vertex wins on
  skewed graphs; thread-per-vertex wins by 2.8× on the road network; a
  degree-adaptive kernel is within 10% of the best everywhere. Against GAP the GPU
  is 1.0–2.6× faster per iteration, but GAP's Gauss–Seidel needs ~40% fewer
  iterations and **reaches convergence first on every graph**.
- **In-place (Gauss–Seidel-style) GPU PageRank** (`gpu_pull_gs`, Phase 12) cuts
  sweeps by up to ~47%, but vertices updated in the same GPU wave stay at Jacobi
  speed. At equal accuracy it is 1.14–1.39× faster than Jacobi on Orkut,
  web-Google and roadNet-CA, and 0.8–0.9× on the other three (report 4.13).
- **Orderings:** random labels cost 2–3× in PageRank. Reverse Cuthill–McKee helps
  PageRank (up to 2.3× on web-Google) but slows bottom-up BFS on social graphs,
  because it delays early exits.

![BFS GTEPS](figures/fig04_bfs_gteps.png)

## Requirements

- Apple Silicon Mac, macOS with Command Line Tools (no Xcode needed: shaders
  are compiled at runtime)
- `brew install libomp`
- Python 3.12/3.13 for analysis scripts (`scripts/setup_python.sh` creates `.venv`)

## Build and test

```sh
./scripts/env_check.sh     # machine facts: chip, cores, memory, disk, toolchain
make                       # builds build/{gconvert,gstats,gbench,gmicro,gtest}
make test                  # unit tests
./build/gmicro --hello     # Metal smoke test + device properties
./build/gmicro --omp       # OpenMP smoke test
./scripts/micro.sh         # Phase 3 microbenchmarks: --launch, --bandwidth, --random
./scripts/setup_python.sh  # analysis venv (numpy pandas matplotlib networkx scipy)
```

Shaders are read from `src/shaders/` at runtime. If you move the build tree,
set `GGPU_SHADER_DIR` to that directory.

## Benchmarking and validation

```sh
# BFS: 64 fixed random sources (largest component); 2 warm-up runs, then per source
# 1 warm-up + 5 timed runs;
# exact check against cpu_serial, rows appended to a CSV
./build/gbench --graph data/soc-LiveJournal1.bin --algo bfs --variant cpu_do \
               --threads 8 --validate --csv results/bfs.csv [--trace trace.csv]
./build/gbench --graph synth:star100k --variant cpu_omp --schedule guided --validate

# PageRank (d = 0.85): fixed 20 iterations (ms/iteration is the metric) or --tol;
# validated against cpu_serial (double): L1 < 1e-4, top-1000 rel. error < 1e-3, sum 1
./build/gbench --graph data/soc-LiveJournal1.bin --algo pr --variant gpu_pull_vector --validate
./build/gbench --graph data/web-Google.bin --algo pr --variant gpu_push --tol 1e-6 --validate
./scripts/bench_pr.sh                          # all PageRank variants x datasets

# Vertex orderings (original | random | degree | rcm), untimed relabeling; results
# are validated in original ids. --locality-csv records mean |u-v| and same-64-block.
./build/gbench --graph data/web-Google.bin --algo pr --variant gpu_pull_adaptive --ordering rcm --validate
./scripts/bench_reorder.sh                     # orderings x datasets (BFS hybrid + PR)

.venv/bin/python tools/validate_networkx.py   # cpu_serial BFS + PageRank vs NetworkX
./scripts/scaling_cpu.sh                       # OpenMP schedule x thread-count sweep
```

BFS variants: `cpu_serial` (reference), `cpu_omp` (level-synchronous,
`--schedule static|dynamic,64|guided`), `cpu_do` (direction-optimizing,
`--alpha 15 --beta 18`), `gpu_vertex` (thread per vertex every level) and
`gpu_naive` (thread per frontier vertex, CAS claim, atomic queue append),
`gpu_binned` (degree bins: simdgroup-cooperative small lists, simdgroup per
medium vertex, threadgroup per large vertex; `--small-max 16 --large-min 256`)
`gpu_edge` (scan of frontier degrees, fixed edges per thread; `--ept 2`) and
`gpu_do` (direction-optimizing: `gpu_edge` top-down, bitmap bottom-up with
early exit; `--alpha 15 --beta 18`, same rule as `cpu_do`) and `hybrid`
(`gpu_do` where levels with work below `--T` (default 1e5) run on the CPU with
OpenMP, `--threads` default 4, directly on the shared buffers; `--T 0` = all
GPU, `--T inf` = all CPU).
GPU variants take `--tg` (threadgroup size, default 256) and also report
kernel-only time (`time_ms_gpu`).

PageRank variants: `cpu_serial` (double, reference), `cpu_omp`, `gpu_push`
(float atomics, simdgroup-aggregated), `gpu_pull_scalar` (thread per vertex),
`gpu_pull_vector` (simdgroup per vertex), `gpu_pull_adaptive` (vertices
binned by in-degree; `--small-max 16 --large-min 256`) and `gpu_pull_gs`
(in-place updates with a per-graph update order: colored or natural; validated
against the converged reference, see `--ref-tol`). `--inplace` makes any pull
variant update in place (experiments).

```sh
./scripts/bench_bfs.sh results/bfs_gpu.csv gpu_vertex gpu_naive gpu_binned gpu_edge   # all datasets, validated
./scripts/sweep_balanced.sh     # gpu_binned thresholds / gpu_edge ept x threadgroup size
./scripts/sweep_do.sh           # gpu_do alpha x beta
./scripts/sweep_hybrid.sh       # hybrid threshold T (0 .. inf) x CPU threads
```

## Reproducing everything

```sh
./scripts/download.sh                    # all six SNAP datasets (~1.3 GB of .bin in data/)
./scripts/setup_python.sh                # .venv for analysis and NetworkX validation
make && make test                        # zero-warning build + 24 test groups
./scripts/run_all.sh                     # full validated matrix -> results/*.csv, then figures/
                                         # (~75 min on an M2; refuses to run on battery or
                                         #  Low Power Mode unless FORCE=1; logs in logs/)
./scripts/run_all.sh pr reorder plots    # or just some stages
./scripts/run_all.sh tuning_balanced tuning_do tuning_tg   # the parameter sweeps (not rerun by default)
./scripts/run_all.sh gs                  # gpu_pull_gs benchmark (~20 min, Phase 12)
./scripts/gap.sh                         # GAP baseline: clones gapbs (pinned) into external/, builds
                                         # with libomp, runs it on our graphs and sources (~6 min)
.venv/bin/python analysis/plots.py       # regenerate the 11 figures (PNG + SVG) from results/
.venv/bin/python analysis/report_tables.py   # markdown tables used in docs/REPORT.md
```

## Layout

```
src/graph   CSR format, converter, transforms, reorderings, generators
src/cpu     CPU BFS / PageRank (serial reference + OpenMP)
src/metal   Metal context, GPU BFS / PageRank, primitives
src/shaders Metal Shading Language kernels (compiled at runtime)
src/bench   gbench (benchmark CLI), gmicro (microbenchmarks)
            (src/graph/export_sg.cpp = gexport: .bin -> GAP .sg)
tests/      assert-based unit tests (make test)
scripts/    environment, data download, full benchmark run
analysis/   plotting; results/ CSVs; figures/ plots; docs/REPORT.md
```

## Data

Datasets come from SNAP and are converted to a compact binary CSR (`data/*.bin`,
not committed; format in `PLAN.md` 4.2).

```sh
./scripts/download.sh                 # all six datasets (~1.4 GB of .bin; .gz deleted after)
./scripts/download.sh web-Google      # or just some of them
./build/gstats data/web-Google.bin    # stats -> results/dataset_stats.csv, degree_hist.csv
./build/gconvert in.txt.gz out.bin --directed|--undirected   # any SNAP edge list
```

The converter streams the gzip file twice (count, then scatter) and never
holds an edge list, so peak memory is about 4 bytes per stored edge.
