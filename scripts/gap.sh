#!/usr/bin/env bash
# Phase 11: GAP Benchmark Suite (Beamer et al.) as an external CPU baseline.
#
#   scripts/gap.sh                   clone (pinned commit) + build + run all datasets
#   scripts/gap.sh web-Google        only these datasets
#   FORCE=1 scripts/gap.sh           run even on battery / Low Power Mode
#
# GAP lives in external/gapbs (gitignored, ~0.6 MB). Its Makefile assumes GCC's
# -fopenmp; Apple clang needs -Xpreprocessor -fopenmp plus Homebrew libomp, so
# we pass CXX_FLAGS on the make command line instead of patching GAP. Flags
# match our own build (-O3 -march=native). Results: results/gap_{bfs,pr}.csv.
set -euo pipefail
cd "$(dirname "$0")/.."

GAP_URL=https://github.com/sbeamer/gapbs.git
GAP_COMMIT=941462c   # master on 2026-10-19
LIBOMP=$(brew --prefix libomp)

if [[ ! -d external/gapbs ]]; then
  mkdir -p external
  git clone -q "$GAP_URL" external/gapbs
fi
git -C external/gapbs checkout -q "$GAP_COMMIT"
make -s -C external/gapbs bfs pr CXX=clang++ \
  CXX_FLAGS="-std=c++11 -O3 -Wall -march=native -mmacosx-version-min=26.0 -Xpreprocessor -fopenmp -I$LIBOMP/include -L$LIBOMP/lib -lomp"
make -s build/gexport

power=$(pmset -g batt | head -1)
lpm=$(pmset -g | awk '/lowpowermode/ {print $2}')
if [[ -z "${FORCE:-}" ]] && { [[ "$power" != *"AC Power"* ]] || [[ "${lpm:-0}" != 0 ]]; }; then
  echo "refusing to run on battery or in Low Power Mode (set FORCE=1 to override)" >&2
  exit 1
fi
.venv/bin/python tools/run_gap.py "$@"
