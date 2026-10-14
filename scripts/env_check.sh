#!/usr/bin/env bash
# Print the machine facts the project depends on (PLAN.md Section 1).
set -u
cd "$(dirname "$0")/.."

echo "== Hardware =="
echo "chip            : $(sysctl -n machdep.cpu.brand_string)"
echo "CPU cores       : $(sysctl -n hw.ncpu) total, $(sysctl -n hw.perflevel0.physicalcpu) performance + $(sysctl -n hw.perflevel1.physicalcpu) efficiency"
echo "memory          : $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB"
echo "page size       : $(sysctl -n hw.pagesize) bytes"
system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model|Total Number of Cores|Metal Support/ {gsub(/^ +/,"",$1); printf "GPU %-12s: %s\n", $1, $2}'
echo "low power mode  : $(pmset -g 2>/dev/null | awk '/lowpowermode/ {print ($2=="1")?"ON (turn off for benchmarks)":"off"}')"
echo "power source    : $(pmset -g batt 2>/dev/null | head -1 | sed "s/.*'\(.*\)'.*/\1/")"

echo; echo "== Disk =="
df -h . | tail -1 | awk '{print "free disk       : " $4 " of " $2}'
du -sh . 2>/dev/null | awk '{print "project size    : " $1}'

echo; echo "== Toolchain =="
echo "macOS           : $(sw_vers -productVersion)"
echo "SDK             : $(xcrun --show-sdk-version 2>/dev/null)"
echo "clang           : $(clang --version | head -1)"
if xcrun -f metal >/dev/null 2>&1; then echo "offline metal   : available"; else echo "offline metal   : not available (shaders compiled at runtime)"; fi
LIBOMP="$(brew --prefix libomp 2>/dev/null)"
if [ -n "$LIBOMP" ] && [ -f "$LIBOMP/lib/libomp.dylib" ]; then
    echo "libomp          : $LIBOMP (version $(basename "$(readlink "$LIBOMP")"))"
else
    echo "libomp          : MISSING (install with: brew install libomp)"
fi
echo "python          : $(python3 --version 2>&1)"
if [ -x .venv/bin/python ]; then echo "venv            : .venv ($(.venv/bin/python --version))"; else echo "venv            : not created (run scripts/setup_python.sh)"; fi
