#!/usr/bin/env bash
# Download SNAP datasets and convert them to data/<name>.bin (PLAN.md 4.1, 4.3).
#   scripts/download.sh            # all six, com-Orkut last
#   scripts/download.sh web-Google # just the named ones
# For each dataset: check free disk, download the .gz, convert (streaming, under
# /usr/bin/time -l for peak memory), verify the .bin loads and validates
# (gstats), then delete the .gz. Existing valid .bin files are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

# name|url|directedness  (order matters: largest last)
DATASETS=(
  "ego-Facebook|https://snap.stanford.edu/data/facebook_combined.txt.gz|undirected"
  "web-Google|https://snap.stanford.edu/data/web-Google.txt.gz|directed"
  "roadNet-CA|https://snap.stanford.edu/data/roadNet-CA.txt.gz|undirected"
  "soc-Pokec|https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz|directed"
  "soc-LiveJournal1|https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz|directed"
  "com-Orkut|https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz|undirected"
)
MIN_FREE_GB=3   # one download (<0.5 GB) + one .bin (<1 GB) + headroom

make -s build/gconvert build/gstats
mkdir -p data/raw results

free_gb() { df -g . | awk 'NR==2 {print $4}'; }

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r name url kind <<< "$entry"
  if [ $# -gt 0 ] && [[ ! " $* " == *" $name "* ]]; then continue; fi
  bin="data/$name.bin"
  gz="data/raw/$name.txt.gz"
  echo "=== $name ==="

  if [ -f "$bin" ] && build/gstats "$bin" --csv "" --hist "" >/dev/null 2>&1; then
    echo "already converted and valid: $bin (skipping)"
    continue
  fi

  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    echo "only $(free_gb) GB free (< $MIN_FREE_GB GB); stopping." >&2
    exit 1
  fi

  if [ ! -f "$gz" ]; then
    echo "downloading $url"
    curl -fL --retry 3 --progress-bar -o "$gz.part" "$url"
    mv "$gz.part" "$gz"
  fi
  ls -lh "$gz" | awk '{print "gz size: " $5}'

  /usr/bin/time -l build/gconvert "$gz" "$bin" --"$kind" 2> "results/convert_time_$name.txt"
  awk '/maximum resident set size/ {printf "peak RSS (/usr/bin/time -l): %.1f MB\n", $1/1048576}' \
      "results/convert_time_$name.txt"

  # Verify before deleting the source: full load + structural validation + stats.
  build/gstats "$bin"
  rm -f "$gz"
  echo
done

du -sh data
