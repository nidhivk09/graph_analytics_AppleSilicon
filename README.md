# graph-gpu: BFS and PageRank on Apple Silicon (Metal)

High-performance breadth-first search and PageRank on an Apple M2 GPU using
Metal compute shaders, compared against OpenMP CPU baselines, with careful
benchmarking on SNAP graphs. See `PLAN.md` for the full design and
`PROGRESS.md` for the running log.

> Status: **Phase 2 (CPU BFS baselines)**. Results and figures will appear here as phases complete.

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
# BFS: 64 fixed random sources (largest component), 2 warm-up + 5 timed runs each,
# exact check against cpu_serial, rows appended to a CSV
./build/gbench --graph data/soc-LiveJournal1.bin --algo bfs --variant cpu_do \
               --threads 8 --validate --csv results/bfs.csv [--trace trace.csv]
./build/gbench --graph synth:star100k --variant cpu_omp --schedule guided --validate

.venv/bin/python tools/validate_networkx.py   # cpu_serial vs NetworkX on small graphs
./scripts/scaling_cpu.sh                       # OpenMP schedule x thread-count sweep
```

BFS variants so far: `cpu_serial` (reference), `cpu_omp` (level-synchronous,
`--schedule static|dynamic,64|guided`), `cpu_do` (direction-optimizing,
`--alpha 15 --beta 18`), `gpu_vertex` (thread per vertex every level) and
`gpu_naive` (thread per frontier vertex, CAS claim, atomic queue append),
`gpu_binned` (degree bins: simdgroup-cooperative small lists, simdgroup per
medium vertex, threadgroup per large vertex; `--small-max 16 --large-min 256`)
`gpu_edge` (scan of frontier degrees, fixed edges per thread; `--ept 2`) and
`gpu_do` (direction-optimizing: `gpu_edge` top-down, bitmap bottom-up with
early exit; `--alpha 15 --beta 18`, same rule as `cpu_do`).
GPU variants take `--tg` (threadgroup size, default 256) and also report
kernel-only time (`time_ms_gpu`).

```sh
./scripts/bench_bfs.sh results/bfs_gpu.csv gpu_vertex gpu_naive gpu_binned gpu_edge   # all datasets, validated
./scripts/sweep_balanced.sh     # gpu_binned thresholds / gpu_edge ept x threadgroup size
./scripts/sweep_do.sh           # gpu_do alpha x beta
```

## Layout

```
src/graph   CSR format, converter, transforms, reorderings, generators
src/cpu     CPU BFS / PageRank (serial reference + OpenMP)
src/metal   Metal context, GPU BFS / PageRank, primitives
src/shaders Metal Shading Language kernels (compiled at runtime)
src/bench   gbench (benchmark CLI), gmicro (microbenchmarks)
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
