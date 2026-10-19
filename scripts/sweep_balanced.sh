#!/usr/bin/env bash
# Phase 5 tuning: gpu_binned degree thresholds and gpu_edge edges-per-thread,
# each against threadgroup size, on LiveJournal and Orkut.
# Reduced protocol (16 sources x 3 trials; same sources everywhere), like the
# Phase 2 scaling sweep. Output: results/sweep_bfs_balanced.csv
set -euo pipefail
cd "$(dirname "$0")/.."
make -s build/gbench
out=results/sweep_bfs_balanced.csv
rm -f "$out"
pmset -g batt | head -1
run() { ./build/gbench --graph "data/$1.bin" --sources 16 --trials 3 --csv "$out" "${@:2}" \
          | grep -oE "median [0-9.]+|GPU-only median [0-9.]+" | head -2 | tr '\n' ' '; echo; }
for d in soc-LiveJournal1 com-Orkut; do
  for s in 8 16 32 64; do for l in 256 1024 4096; do
    printf '%-17s binned small<%-3s large>=%-5s tg 256  ' $d $s $l
    run $d --variant gpu_binned --small-max $s --large-min $l --tg 256
  done; done
  for tg in 64 128 512 1024; do
    printf '%-17s binned small<16  large>=512   tg %-4s ' $d $tg
    run $d --variant gpu_binned --small-max 16 --large-min 512 --tg $tg
  done
  for e in 1 2 4 8 16; do for tg in 64 128 256 512 1024; do
    printf '%-17s edge ept %-3s tg %-4s ' $d $e $tg
    run $d --variant gpu_edge --ept $e --tg $tg
  done; done
done
