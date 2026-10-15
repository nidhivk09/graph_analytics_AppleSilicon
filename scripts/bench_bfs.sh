#!/usr/bin/env bash
# Benchmark + validate BFS variants on every dataset (full protocol: 64
# sources x (2 warm-up + 5 trials), exact check against cpu_serial).
#   scripts/bench_bfs.sh results/bfs_gpu.csv gpu_vertex gpu_naive
# Optional per-level traces (4 sources, untimed) for LiveJournal and roadNet-CA:
#   TRACE=results/trace_bfs.csv scripts/bench_bfs.sh ...
set -euo pipefail
cd "$(dirname "$0")/.."
csv=$1; shift
make -s build/gbench
pmset -g batt | head -1
datasets="ego-Facebook web-Google roadNet-CA soc-Pokec soc-LiveJournal1 com-Orkut"
for v in "$@"; do
  for d in $datasets; do
    printf '%-17s %-11s ' "$d" "$v"
    ./build/gbench --graph "data/$d.bin" --variant "$v" --validate --csv "$csv" ${EXTRA:-} | grep -E "time ms|FAILED"
  done
  if [ -n "${TRACE:-}" ]; then
    for d in soc-LiveJournal1 roadNet-CA; do
      ./build/gbench --graph "data/$d.bin" --variant "$v" --sources 4 --warmup 1 --trials 0 \
                     --trace "$TRACE" ${EXTRA:-} > /dev/null
    done
  fi
done
