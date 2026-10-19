#!/usr/bin/env bash
# Phase 10: the full benchmark matrix, every timed run validated (gbench
# --validate checks each trial against the cpu_serial reference, outside the
# timed region; validation failures abort). Regenerates every results/*.csv
# used by analysis/plots.py, then the figures.
#
#   scripts/run_all.sh              all stages
#   scripts/run_all.sh pr reorder   only these stages
#   FORCE=1 scripts/run_all.sh      run even on battery / Low Power Mode
#
# Stages (estimates from Phase 2-9 runs on an M2 / 8 GB):
STAGES=(test micro stats bfs_cpu scaling bfs_gpu star hybrid_sweep pr reorder networkx plots)
est() {  # minutes (bash 3.2 on macOS has no associative arrays)
  case $1 in
    test) echo 2;; micro) echo 1;; stats) echo 3;; bfs_cpu) echo 25;; scaling) echo 30;;
    bfs_gpu) echo 30;; star) echo 1;; hybrid_sweep) echo 12;; pr) echo 15;; reorder) echo 32;;
    networkx) echo 3;; plots) echo 1;; tuning_balanced) echo 20;; tuning_do) echo 10;;
    tuning_tg) echo 12;; *) echo 10;;
  esac
}
# Tuning sweeps (Phase 5/6/10) are not rerun by default: their CSVs record how
# the defaults were chosen. Run them explicitly: tuning_balanced tuning_do tuning_tg.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -gt 0 ]; then STAGES=("$@"); fi
total=0
echo "run_all: stages and estimated runtime (minutes):"
for s in "${STAGES[@]}"; do
  m=$(est "$s"); total=$((total + m))
  printf '  %-15s ~%3d min\n' "$s" "$m"
done
printf '  %-15s ~%3d min (%.1f h)\n' total $total "$(echo "$total/60" | bc -l)"

power=$(pmset -g batt | head -1)
lpm=$(pmset -g | awk '/lowpowermode/ {print $2}')
echo "power: $power; lowpowermode = ${lpm:-?}"
echo "Reminder: plugged in, Low Power Mode off, other apps closed (methodology, PLAN.md 8)."
if [[ -z "${FORCE:-}" ]] && { [[ "$power" != *"AC Power"* ]] || [[ "${lpm:-0}" != 0 ]]; }; then
  echo "refusing to run on battery or in Low Power Mode (set FORCE=1 to override)" >&2
  exit 1
fi

make -s
mkdir -p results figures logs
log=logs/run_all_$(date +%Y%m%d_%H%M%S).log
echo "logging to $log"
DATASETS="ego-Facebook web-Google roadNet-CA soc-Pokec soc-LiveJournal1 com-Orkut"

stage() {
  local s=$1 t0=$SECONDS
  echo "=== $s ($(date +%H:%M:%S)) ===" | tee -a "$log"
  case $s in
    test)    make test 2>&1 | grep -E "FAIL|^    |passed" ;;  # failing tests + their CHECK lines
    micro)   ./scripts/micro.sh ;;
    stats)   rm -f results/dataset_stats.csv results/degree_hist.csv
             for d in $DATASETS; do ./build/gstats data/$d.bin; done ;;
    bfs_cpu) rm -f results/bfs_cpu.csv
             EXTRA="--threads 8" ./scripts/bench_bfs.sh results/bfs_cpu.csv cpu_serial cpu_omp cpu_do ;;
    scaling) ./scripts/scaling_cpu.sh ;;
    bfs_gpu) rm -f results/bfs_gpu.csv results/trace_bfs.csv
             TRACE=results/trace_bfs.csv ./scripts/bench_bfs.sh results/bfs_gpu.csv \
               gpu_vertex gpu_naive gpu_binned gpu_edge gpu_do hybrid ;;
    star)    rm -f results/bfs_star.csv
             for v in cpu_serial cpu_omp cpu_do gpu_vertex gpu_naive gpu_binned gpu_edge gpu_do hybrid; do
               printf 'star100k %-11s ' $v
               ./build/gbench --graph synth:star100k --variant $v --validate --csv results/bfs_star.csv \
                 | grep -E "time ms|FAILED"
             done ;;
    hybrid_sweep) ./scripts/sweep_hybrid.sh ;;
    pr)      rm -f results/pr.csv results/pr_converge.csv; ./scripts/bench_pr.sh ;;
    reorder) rm -f results/reorder_locality.csv; ./scripts/bench_reorder.sh ;;
    networkx) .venv/bin/python tools/validate_networkx.py | tail -1 ;;
    plots)   .venv/bin/python analysis/plots.py ;;
    tuning_balanced) ./scripts/sweep_balanced.sh ;;
    tuning_do)       ./scripts/sweep_do.sh ;;
    tuning_tg)       ./scripts/sweep_tg.sh ;;
    *) echo "unknown stage '$s'" >&2; return 2 ;;
  esac 2>&1 | tee -a "$log"
  # A validation failure inside a stage makes gbench exit non-zero; with
  # pipefail that fails the pipeline, and set -e stops run_all here.
  echo "--- $s done in $(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s" | tee -a "$log"
}

for s in "${STAGES[@]}"; do stage "$s"; done
if grep -q "FAILED" "$log"; then echo "run_all: FAILURES found in $log" >&2; exit 1; fi
echo "run_all: all stages done; log $log"
