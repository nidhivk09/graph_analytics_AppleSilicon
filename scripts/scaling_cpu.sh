#!/usr/bin/env bash
# Phase 2 P-core/E-core experiment: cpu_omp with static / dynamic,64 / guided
# at 1,2,4,6,8 threads (plus cpu_do and cpu_serial for context) on
# soc-LiveJournal1 and roadNet-CA. -> results/scaling_cpu_bfs.csv
#
# 16 sources x 3 trials (+2 warm-up) per configuration: the full 64 x 5 would
# take over an hour at 1 thread on LiveJournal. Sources are identical across
# configurations (fixed seed), so configurations are directly comparable.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/scaling_cpu_bfs.csv}
COMMON="--algo bfs --sources ${SOURCES:-16} --trials ${TRIALS:-3} --warmup 2 --validate --csv $OUT"

echo "Reminder: plugged in, Low Power Mode off, other apps closed."
pmset -g | awk '/lowpowermode/ {print "lowpowermode =", $2}'
make -s build/gbench
rm -f "$OUT"

for d in roadNet-CA soc-LiveJournal1; do
  g=data/$d.bin
  build/gbench --graph $g --variant cpu_serial $COMMON | tail -1 | sed "s/^/$d cpu_serial: /"
  for t in 1 2 4 6 8; do
    for s in static dynamic,64 guided; do
      build/gbench --graph $g --variant cpu_omp --threads $t --schedule $s $COMMON | tail -1 \
        | sed "s/^/$d cpu_omp t=$t $s: /"
    done
    build/gbench --graph $g --variant cpu_do --threads $t $COMMON | tail -1 | sed "s/^/$d cpu_do t=$t: /"
  done
done
