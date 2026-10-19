#!/usr/bin/env bash
# Phase 10: threadgroup-size sweep for the main kernels.
#   BFS gpu_do (top-down edge expand + bottom-up): LiveJournal, Pokec, Orkut
#   PR gpu_pull_vector / gpu_pull_adaptive: LiveJournal, Orkut, roadNet-CA, web-Google
# Reduced protocol (BFS 16 sources x 3 trials; PR 20 iterations, 1+3 runs).
# -> results/sweep_tg.csv
set -euo pipefail
cd "$(dirname "$0")/.."
make -s build/gbench
out=results/sweep_tg.csv
rm -f "$out"
pmset -g batt | head -1
for d in soc-LiveJournal1 soc-Pokec com-Orkut; do
  for tg in 64 128 256 512 1024; do
    printf '%-17s gpu_do            tg %-4s ' $d $tg
    ./build/gbench --graph "data/$d.bin" --variant gpu_do --tg $tg --sources 16 --trials 3 --csv "$out" \
      | grep -oE "median [0-9.]+" | head -1
  done
done
for d in soc-LiveJournal1 com-Orkut roadNet-CA web-Google; do
  for v in gpu_pull_vector gpu_pull_adaptive; do
    for tg in 64 128 256 512 1024; do
      printf '%-17s %-17s tg %-4s ' $d $v $tg
      ./build/gbench --graph "data/$d.bin" --algo pr --variant $v --tg $tg --warmup 1 --trials 3 --csv "$out" \
        | grep -oE "[0-9.]+ ms/iteration"
    done
  done
done
