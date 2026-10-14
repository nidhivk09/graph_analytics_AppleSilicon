# graph-gpu: BFS and PageRank on Apple Silicon (Metal)

High-performance breadth-first search and PageRank on an Apple M2 GPU using
Metal compute shaders, compared against OpenMP CPU baselines, with careful
benchmarking on SNAP graphs. See `PLAN.md` for the full design and
`PROGRESS.md` for the running log.

> Status: **Phase 1 (graph I/O + datasets)**. Results and figures will appear here as phases complete.

## Requirements

- Apple Silicon Mac, macOS with Command Line Tools (no Xcode needed: shaders
  are compiled at runtime)
- `brew install libomp`
- Python 3.12/3.13 for analysis scripts (`scripts/setup_python.sh` creates `.venv`)

## Build and test

```sh
./scripts/env_check.sh     # machine facts: chip, cores, memory, disk, toolchain
make                       # builds build/{gconvert,gstats,gmicro,gtest}
make test                  # unit tests
./build/gmicro --hello     # Metal smoke test + device properties
./build/gmicro --omp       # OpenMP smoke test
./scripts/setup_python.sh  # analysis venv (numpy pandas matplotlib networkx scipy)
```

Shaders are read from `src/shaders/` at runtime. If you move the build tree,
set `GGPU_SHADER_DIR` to that directory.

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
