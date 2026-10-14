#!/usr/bin/env bash
# Create .venv with the analysis-only Python dependencies (PLAN.md Section 2).
# Override the interpreter with PYTHON=/path/to/python3.
set -euo pipefail
cd "$(dirname "$0")/.."

# Homebrew python@3.14 on this machine has a broken pyexpat (links against an
# older system libexpat), which breaks pip. Pick the first interpreter that can
# import pyexpat and report the macOS version.
if [ -z "${PYTHON:-}" ]; then
    for cand in python3.13 python3.12 python3; do
        if command -v "$cand" >/dev/null 2>&1 &&
           "$cand" -c "import pyexpat, platform; assert platform.mac_ver()[0]" 2>/dev/null; then
            PYTHON="$cand"; break
        fi
    done
fi
: "${PYTHON:?no working python3 found (need pyexpat); set PYTHON=...}"
echo "using $PYTHON ($("$PYTHON" --version))"

rm -rf .venv
"$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet numpy pandas matplotlib networkx scipy
.venv/bin/python -c "import numpy, pandas, matplotlib, networkx, scipy; print('venv OK: numpy', numpy.__version__, '| pandas', pandas.__version__, '| matplotlib', matplotlib.__version__, '| networkx', networkx.__version__, '| scipy', scipy.__version__)"
du -sh .venv
