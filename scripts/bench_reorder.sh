#!/usr/bin/env bash
# Phase 9: vertex orderings x datasets. Full protocol, every run validated
# against the ORIGINAL-label reference after mapping results back.
#   BFS: hybrid (best GPU BFS, T=1e5)          -> results/reorder_bfs.csv
#   PR:  gpu_pull_vector, gpu_pull_adaptive    -> results/reorder_pr.csv
#   locality metrics                           -> results/reorder_locality.csv
#   Phase 5 question (access order): gpu_vertex vs gpu_edge on LiveJournal,
#   original vs random, with traces            -> results/trace_reorder_access.csv
set -euo pipefail
cd "$(dirname "$0")/.."
make -s build/gbench
pmset -g batt | head -1
rm -f results/reorder_bfs.csv results/reorder_pr.csv results/trace_reorder_access.csv
datasets="ego-Facebook web-Google roadNet-CA soc-Pokec soc-LiveJournal1 com-Orkut"
for d in $datasets; do
  for o in original random degree rcm; do
    printf '%-17s %-8s hybrid            ' $d $o
    ./build/gbench --graph "data/$d.bin" --variant hybrid --ordering $o --validate \
                   --csv results/reorder_bfs.csv --locality-csv results/reorder_locality.csv \
      | grep -E "time ms|FAILED" | sed -E 's/time ms: //'
    for v in gpu_pull_vector gpu_pull_adaptive; do
      printf '%-17s %-8s %-17s ' $d $o $v
      ./build/gbench --graph "data/$d.bin" --algo pr --variant $v --ordering $o --validate \
                     --csv results/reorder_pr.csv | grep -E "time ms|FAILED" | sed -E 's/time ms: //; s/\(ref [^)]*\)//'
    done
  done
done
for o in original random; do
  for v in gpu_vertex gpu_edge; do
    printf 'soc-LiveJournal1  %-8s %-17s ' $o $v
    ./build/gbench --graph data/soc-LiveJournal1.bin --variant $v --ordering $o --sources 4 --trials 3 \
                   --validate --trace results/trace_reorder_access.csv | grep -E "time ms|FAILED" | sed -E 's/time ms: //'
  done
done
