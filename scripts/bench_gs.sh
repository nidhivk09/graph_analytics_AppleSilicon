#!/usr/bin/env bash
# Phase 12: gpu_pull_gs (in-place, Gauss-Seidel-style PageRank) on every dataset.
#   fixed:     --iters 20, 2 warm-up + 5 trials                -> results/pr_gs.csv
#   equal accuracy: GS's per-sweep change understates its error more than
#              Jacobi's does (in-wave structures converge at the Jacobi rate
#              while the rest is faster), so at equal tol GS stops less
#              accurate. Try tol 1e-6, 5e-7, ... and keep the first run whose
#              distance to the converged reference is no larger than
#              Jacobi's at 1e-6, in BOTH L1 and top-1000 max relative error
#              (strict: no PLAN.md 7 floors). Runs that fail gbench's own
#              validation are reported and the search continues.
#                                                             -> results/pr_gs_equal.csv
#   ablation:  in-place updates with other update orders (degree bins =
#              gpu_pull_adaptive, natural = gpu_pull_vector), tol 1e-6,
#              iteration counts only (not validated)            -> results/pr_gs_ablation.csv
# Every gpu_pull_gs run is validated (cpu::gsNoWorseThanJacobi, see gbench).
set -euo pipefail
cd "$(dirname "$0")/.."
make -s build/gbench
pmset -g batt | head -1
datasets="${DATASETS:-ego-Facebook web-Google roadNet-CA soc-Pokec soc-LiveJournal1 com-Orkut}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
num='[0-9.e+-]+'
append_csv() {  # append $1's rows to $2, keeping one header
  if [ -f "$2" ]; then tail -n +2 "$1" >> "$2"; else cp "$1" "$2"; fi
}

for d in $datasets; do
  g="data/$d.bin"
  printf '%-17s fixed    ' "$d"
  ./build/gbench --graph "$g" --algo pr --variant gpu_pull_gs --iters 20 --validate \
                 --csv results/pr_gs.csv | grep -E "time ms|FAILED" | sed 's/time ms: //; s/| eff.*//'

  found=""
  for t in 1e-6 5e-7 2.5e-7 1.25e-7 6.25e-8 3.125e-8; do
    rm -f "$tmp/run.csv"
    if ! out=$(./build/gbench --graph "$g" --algo pr --variant gpu_pull_gs --tol "$t" --ref-tol 1e-6 \
                              --warmup 1 --trials 3 --validate --csv "$tmp/run.csv" 2>&1); then
      printf '%-17s tol %-8s less accurate than Jacobi@1e-6: %s\n' "$d" "$t" \
             "$(echo "$out" | grep FAILED | sed 's/.*trial 0: //')"
      continue
    fi
    out=$(echo "$out" | grep "time ms")
    # worst L1, top-1000 rel (GS) and the same for Jacobi at 1e-6, vs the converged reference
    read -r gl1 grel jl1 jrel < <(echo "$out" | sed -E "s/.*worst L1 ($num), top-1000 rel ($num).*Jacobi ref [0-9]+ iters: L1 ($num), top-1000 rel ($num).*/\1 \2 \3 \4/")
    iters=$(echo "$out" | sed -E 's/.*\| ([0-9]+) iterations.*/\1/')
    med=$(echo "$out" | sed -E "s/.*median ($num).*/\1/")
    printf '%-17s tol %-8s %3s iterations %9s ms | GS L1 %s rel %s vs Jacobi@1e-6 L1 %s rel %s\n' \
           "$d" "$t" "$iters" "$med" "$gl1" "$grel" "$jl1" "$jrel"
    if awk -v a="$gl1" -v b="$jl1" -v c="$grel" -v e="$jrel" 'BEGIN { exit !(a <= b && c <= e) }'; then
      found=$t
      append_csv "$tmp/run.csv" results/pr_gs_equal.csv
      break
    fi
  done
  echo "$d: equal-accuracy tol = ${found:-not reached}"

  for v in gpu_pull_adaptive gpu_pull_vector; do
    printf '%-17s ablation %-18s ' "$d" "$v --inplace"
    ./build/gbench --graph "$g" --algo pr --variant "$v" --inplace --tol 1e-6 --warmup 1 --trials 2 \
                   --csv results/pr_gs_ablation.csv | grep -E "time ms" | sed -E 's/.*\| ([0-9]+ iterations, [0-9.]+ ms\/iteration).*/\1/'
  done
done
