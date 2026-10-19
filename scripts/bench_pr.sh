#!/usr/bin/env bash
# Phase 8: PageRank on every dataset, every variant, validated against
# cpu_serial (double).
#   fixed mode: --iters 20, 2 warm-up + 5 trials   -> results/pr.csv
#   tol mode:   --tol 1e-6, 1 warm-up + 3 trials   -> results/pr_converge.csv
set -euo pipefail
cd "$(dirname "$0")/.."
make -s build/gbench
pmset -g batt | head -1
datasets="ego-Facebook web-Google roadNet-CA soc-Pokec soc-LiveJournal1 com-Orkut"
variants="${VARIANTS:-cpu_serial cpu_omp gpu_push gpu_pull_scalar gpu_pull_vector gpu_pull_adaptive}"
for d in $datasets; do
  for v in $variants; do
    printf '%-17s %-18s ' "$d" "$v"
    ./build/gbench --graph "data/$d.bin" --algo pr --variant "$v" --iters 20 --validate \
                   --csv results/pr.csv ${EXTRA:-} | grep -E "time ms|FAILED" | sed 's/time ms: //'
  done
done
for d in $datasets; do
  for v in $variants; do
    printf '%-17s %-18s tol ' "$d" "$v"
    ./build/gbench --graph "data/$d.bin" --algo pr --variant "$v" --tol 1e-6 --warmup 1 --trials 3 \
                   --validate --csv results/pr_converge.csv ${EXTRA:-} | grep -E "time ms|FAILED" | sed 's/time ms: //'
  done
done
