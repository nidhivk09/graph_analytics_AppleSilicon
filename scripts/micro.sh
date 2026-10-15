#!/usr/bin/env bash
# Phase 3: run all microbenchmarks -> results/micro_{bandwidth,launch,random}.csv
# Peak memory ~2.3 GB (2 GB table + 128 MB index array). Plug in, Low Power
# Mode off, close other apps first: these numbers are the reference points
# for every later bandwidth/latency argument.
set -euo pipefail
cd "$(dirname "$0")/.."
make -s build/gmicro
mkdir -p results
pmset -g batt | head -1
./build/gmicro --launch
./build/gmicro --bandwidth
./build/gmicro --random
