#!/usr/bin/env bash
# Phase 6: gpu_do alpha/beta sweep on LiveJournal and Pokec.
# Reduced protocol (16 sources x 3 trials, same sources). -> results/sweep_bfs_do.csv
set -euo pipefail
cd "$(dirname "$0")/.."
make -s build/gbench
out=results/sweep_bfs_do.csv
rm -f "$out"
pmset -g batt | head -1
for d in soc-LiveJournal1 soc-Pokec; do
  for a in 5 10 15 20 30; do for b in 10 18 24; do
    printf '%-17s alpha %-3s beta %-3s ' $d $a $b
    ./build/gbench --graph "data/$d.bin" --variant gpu_do --alpha $a --beta $b --sources 16 --trials 3 \
                   --csv "$out" | grep -oE "median [0-9.]+|GPU-only median [0-9.]+" | head -2 | tr '\n' ' '
    echo
  done; done
done
