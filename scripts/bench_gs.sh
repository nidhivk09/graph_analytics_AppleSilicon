#!/usr/bin/env bash
# Phases 12-13: gpu_pull_gs (in-place, Gauss-Seidel-style PageRank) on every dataset.
#   fixed:     --iters 20, 2 warm-up + 5 trials                  -> results/pr_gs.csv
#   equal accuracy: at equal tol GS can stop less accurate than Jacobi (its
#              per-sweep change understates its error more), so every
#              candidate stopping rule below is run (1 warm-up + 3 trials,
#              validated) and judged against Jacobi at tol 1e-6 by distance to
#              the converged reference: it passes if BOTH its L1 and its
#              top-1000 max relative error are no larger (strict: no PLAN.md 7
#              floors). The fastest passing rule is kept.
#              all candidates                                   -> results/pr_gs_search.csv
#              fastest passing run's trials                     -> results/pr_gs_equal.csv
#              Rules: L1 tol 1e-6 ... 3.1e-8 (Phase 12), and L1 tol 1e-6 plus
#              the per-vertex rule --tol-rel (Phase 13).
#   pairs:     tol 1e-6 with mutual-pair deferral off (K=0) vs on (K=2),
#              distance to the converged reference              -> results/pr_gs_pairs.csv
#   ablation:  in-place updates with other orders (degree bins =
#              gpu_pull_adaptive, natural = gpu_pull_vector), tol 1e-6,
#              iteration counts only (not validated)            -> results/pr_gs_ablation.csv
# Every gpu_pull_gs run is validated (cpu::gsNoWorseThanJacobi, see gbench).
set -euo pipefail
cd "$(dirname "$0")/.."
make -s build/gbench
pmset -g batt | head -1
datasets="${DATASETS:-ego-Facebook web-Google roadNet-CA soc-Pokec soc-LiveJournal1 com-Orkut}"
rules=("--tol 1e-6" "--tol 5e-7" "--tol 2.5e-7" "--tol 1.25e-7" "--tol 6.25e-8" "--tol 3.125e-8"
       "--tol 1e-6 --tol-rel 1e-4" "--tol 1e-6 --tol-rel 5e-5" "--tol 1e-6 --tol-rel 2.5e-5")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
num='[0-9.e+-]+'
append_csv() {  # append $1's rows to $2, keeping one header
  if [ -f "$2" ]; then tail -n +2 "$1" >> "$2"; else cp "$1" "$2"; fi
}
[ -f results/pr_gs_search.csv ] || echo "dataset,rule,gs_pairs,iters,median_ms,gs_l1,gs_rel,jacobi_l1,jacobi_rel,pass" > results/pr_gs_search.csv
[ -f results/pr_gs_pairs.csv ] || echo "dataset,gs_pairs,deferred,iters,gs_l1,gs_rel,jacobi_l1,jacobi_rel" > results/pr_gs_pairs.csv

# One validated tol-mode run. Prints "iters median gs_l1 gs_rel jac_l1 jac_rel deferred"
# (accuracy vs the converged reference; works whether gbench's own check passed or not).
run_tol() {  # dataset csv args...
  local g="data/$1.bin" csvout="$2"; shift 2
  local out
  out=$(./build/gbench --graph "$g" --algo pr --variant gpu_pull_gs --ref-tol 1e-6 --warmup 1 --trials 3 \
                       --validate --csv "$csvout" "$@" 2>&1 || true)
  local deferred; deferred=$(echo "$out" | grep -oE "deferred=[0-9]+" | head -1 | cut -d= -f2 || true)
  if echo "$out" | grep -q "time ms"; then
    echo "$out" | grep "time ms" | sed -E "s/.*\| ([0-9]+) iterations.*/\1/" | tr '\n' ' '
    echo "$out" | grep "time ms" | sed -E "s/.*median ($num).*/\1/" | tr '\n' ' '
    echo "$out" | grep "time ms" | sed -E "s/.*worst L1 ($num), top-1000 rel ($num).*Jacobi ref [0-9]+ iters: L1 ($num), top-1000 rel ($num).*/\1 \2 \3 \4/" | tr '\n' ' '
  else  # gbench's validation failed: take the failing trial's numbers
    echo -n "- - "
    echo "$out" | sed -nE "s/.*FAILED.*: L1 ($num), max rel \(top-1000\) ($num),.*/\1 \2/p" | head -1 | tr '\n' ' '
    echo "$out" | sed -nE "s/.*Jacobi reference: L1 ($num), max rel ($num),.*/\1 \2/p" | head -1 | tr '\n' ' '
  fi
  echo "${deferred:-0}"
}

for d in $datasets; do
  printf '%-17s fixed    ' "$d"
  ./build/gbench --graph "data/$d.bin" --algo pr --variant gpu_pull_gs --iters 20 --validate \
                 --csv results/pr_gs.csv | grep -E "time ms|FAILED" | sed 's/time ms: //; s/| eff.*//'

  best="" bestms=""
  for rule in "${rules[@]}"; do
    rm -f "$tmp/run.csv"
    # shellcheck disable=SC2086  # the rule is several words on purpose
    read -r it ms gl1 grel jl1 jrel deferred < <(run_tol "$d" "$tmp/run.csv" $rule)
    pass=0
    if [ "$it" != "-" ] && awk -v a="$gl1" -v b="$jl1" -v c="$grel" -v e="$jrel" 'BEGIN { exit !(a <= b && c <= e) }'; then
      pass=1
      if [ -z "$bestms" ] || awk -v a="$ms" -v b="$bestms" 'BEGIN { exit !(a < b) }'; then
        best=$rule bestms=$ms
        cp "$tmp/run.csv" "$tmp/best.csv"
      fi
    fi
    echo "$d,$rule,2,$it,$ms,$gl1,$grel,$jl1,$jrel,$pass" >> results/pr_gs_search.csv
    printf '%-17s %-28s %3s sweeps %9s ms | GS L1 %s rel %s vs Jacobi@1e-6 L1 %s rel %s%s\n' \
           "$d" "$rule" "$it" "$ms" "$gl1" "$grel" "$jl1" "$jrel" "$([ $pass = 1 ] && echo '  OK')"
  done
  if [ -n "$best" ]; then append_csv "$tmp/best.csv" results/pr_gs_equal.csv; fi
  echo "$d: fastest equal-accuracy rule = ${best:-none} (${bestms:-} ms)"

  for K in 0 2; do
    read -r it ms gl1 grel jl1 jrel deferred < <(run_tol "$d" /dev/null --tol 1e-6 --gs-pairs "$K")
    echo "$d,$K,$deferred,$it,$gl1,$grel,$jl1,$jrel" >> results/pr_gs_pairs.csv
    printf '%-17s pairs K=%s (deferred %s): top-1000 rel %s, L1 %s\n' "$d" "$K" "$deferred" "$grel" "$gl1"
  done

  for v in gpu_pull_adaptive gpu_pull_vector; do
    printf '%-17s ablation %-18s ' "$d" "$v --inplace"
    ./build/gbench --graph "data/$d.bin" --algo pr --variant "$v" --inplace --tol 1e-6 --warmup 1 --trials 2 \
                   --csv results/pr_gs_ablation.csv | grep -E "time ms" | sed -E 's/.*\| ([0-9]+ iterations, [0-9.]+ ms\/iteration).*/\1/'
  done
done
