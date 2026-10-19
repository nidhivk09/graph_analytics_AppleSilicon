#!/usr/bin/env bash
# Phase 7: hybrid threshold sweep. T = 0 (all GPU) ... inf (all CPU), log
# scale, CPU levels with 4 (P-cores only, Phase 2's best for small levels) and
# 8 threads. Reduced protocol (16 sources x 3 trials, same sources).
# -> results/sweep_hybrid.csv; traces at the chosen T are produced separately.
set -euo pipefail
cd "$(dirname "$0")/.."
make -s build/gbench
out=results/sweep_hybrid.csv
rm -f "$out"
pmset -g batt | head -1
for d in roadNet-CA soc-LiveJournal1 web-Google; do
  for th in 4 8; do
    for T in 0 100 1000 3000 10000 30000 100000 300000 1000000 3000000 10000000 inf; do
      printf '%-17s threads %s T %-9s ' $d $th $T
      ./build/gbench --graph "data/$d.bin" --variant hybrid --T $T --threads $th --sources 16 --trials 3 \
                     --csv "$out" | grep -oE "median [0-9.]+|GPU-only median [0-9.]+" | head -2 | tr '\n' ' '
      echo
    done
  done
done
