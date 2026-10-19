#!/usr/bin/env python3
"""Cross-check the C++ reference implementations against NetworkX (PLAN.md 7).

This validates the *reference itself*: every other variant is checked against
cpu_serial, so cpu_serial is checked against an independent implementation.

  .venv/bin/python tools/validate_networkx.py            # all graphs
  .venv/bin/python tools/validate_networkx.py --graphs synth:path1000 data/web-Google.bin

BFS: `gbench --variant cpu_serial --dump-dist` vs
networkx.single_source_shortest_path_length on the undirected (symmetrized)
graph, for several sources per graph; distances must match exactly.
PageRank: `gbench --algo pr --variant cpu_serial --tol 1e-12 --dump-ranks` vs
networkx.pagerank(alpha=0.85) on the directed graph. networkx's `tol` is per
node (it stops when the L1 change < N * tol), so PLAN.md's tol=1e-10 would
allow an L1 error of ~1e-4 on web-Google; we pass tol = 1e-10 / N (L1 change
< 1e-10) and run cpu_serial with --tol 1e-10, so both sides are within ~1e-9
of the fixed point, far below the comparison threshold. (1e-12 is not
reachable: on star100k double-precision power iteration ends in a rounding
2-cycle with L1 change 1.35e-11, from the hub's 100k-term sum.): L1 < 1e-6 and max relative error on the
top-1000 vertices < 1e-4 (stricter than PLAN.md 7's GPU tolerances, since
both sides are double precision).
Writes results/validate_networkx.csv.
"""
import argparse
import csv
import os
import resource
import subprocess
import sys
import tempfile
import time

import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GBENCH = os.path.join(ROOT, "build", "gbench")
UNREACHED = 0xFFFFFFFF
SYNTHETIC = ["path1000", "star100k", "complete200", "two_components", "empty1000",
             "grid100x100", "rmat12", "rmat12_directed"]
DEFAULT_GRAPHS = ["synth:" + s for s in SYNTHETIC] + ["data/ego-Facebook.bin", "data/web-Google.bin"]


def load_bin(path):
    """Read the GGPUCSR1 format (PLAN.md 4.2) -> (n, flags, row_ptr, col_idx)."""
    with open(path, "rb") as f:
        header = f.read(32)
        if header[:8] != b"GGPUCSR1":
            raise ValueError(f"{path}: bad magic")
        version, flags = np.frombuffer(header[8:16], dtype="<u4")
        n, m = np.frombuffer(header[16:32], dtype="<u8")
        if version != 1:
            raise ValueError(f"{path}: unsupported version {version}")
        row_ptr = np.fromfile(f, dtype="<u4", count=int(n) + 1)
        col_idx = np.fromfile(f, dtype="<u4", count=int(m))
    return int(n), int(flags), row_ptr, col_idx


def to_networkx_undirected(n, row_ptr, col_idx):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    src = np.repeat(np.arange(n, dtype=np.uint32), np.diff(row_ptr).astype(np.int64))
    g.add_edges_from(zip(src.tolist(), col_idx.tolist()))
    return g


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr}")
    return r.stdout


def pick_sources(n, row_ptr, k, seed=7):
    """Deterministic sources: vertex 0, the max-degree vertex, then random ones."""
    deg = np.diff(row_ptr)
    rng = np.random.default_rng(seed)
    out = [0, int(np.argmax(deg))] if n else []
    while len(out) < min(k, n):
        out.append(int(rng.integers(n)))
    return list(dict.fromkeys(out))  # dedupe, keep order


def to_networkx_directed(n, row_ptr, col_idx):
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    src = np.repeat(np.arange(n, dtype=np.uint32), np.diff(row_ptr).astype(np.int64))
    g.add_edges_from(zip(src.tolist(), col_idx.tolist()))
    return g


def check_pr(spec, n, row_ptr, col_idx, tmp):
    dump = os.path.join(tmp, "ranks.f64")
    run([GBENCH, "--graph", spec, "--algo", "pr", "--variant", "cpu_serial", "--tol", "1e-10",
         "--iters", "10000", "--warmup", "0", "--trials", "0", "--dump-ranks", dump])
    ours = np.fromfile(dump, dtype="<f8")
    g = to_networkx_directed(n, row_ptr, col_idx)
    pr = nx.pagerank(g, alpha=0.85, tol=1e-10 / max(n, 1), max_iter=10000)
    expect = np.array([pr[v] for v in range(n)])
    del g
    l1 = float(np.abs(ours - expect).sum())
    top = np.argsort(-expect)[:1000]
    rel = float(np.max(np.abs(ours[top] - expect[top]) / expect[top])) if n else 0.0
    ok = l1 < 1e-6 and rel < 1e-4
    print(f"  pagerank: L1 {l1:.2e}, max rel (top-1000) {rel:.2e} -> {'OK' if ok else 'FAIL'}")
    return [{"graph": spec, "algo": "pr", "source": "", "n": n, "reached": "",
             "mismatches": f"L1={l1:.3e} rel={rel:.3e}", "status": "OK" if ok else "FAIL"}]


def check_bfs(spec, n, row_ptr, col_idx, g, sources, tmp):
    rows = []
    for s in sources:
        dump = os.path.join(tmp, "dist.u32")
        run([GBENCH, "--graph", spec, "--algo", "bfs", "--variant", "cpu_serial", "--source", str(s),
             "--warmup", "0", "--trials", "0", "--dump-dist", dump])
        ours = np.fromfile(dump, dtype="<u4")
        expect = np.full(n, UNREACHED, dtype=np.uint64)
        for v, d in nx.single_source_shortest_path_length(g, s).items():
            expect[v] = d
        mismatches = int(np.count_nonzero(ours.astype(np.uint64) != expect))
        reached = int(np.count_nonzero(expect != UNREACHED))
        rows.append({"graph": spec, "algo": "bfs", "source": s, "n": n, "reached": reached,
                     "mismatches": mismatches, "status": "OK" if mismatches == 0 else "FAIL"})
        print(f"  bfs source {s:>8}: reached {reached:>8}, mismatches {mismatches} -> {rows[-1]['status']}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", nargs="*", default=DEFAULT_GRAPHS)
    ap.add_argument("--sources", type=int, default=4)
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "validate_networkx.csv"))
    args = ap.parse_args()
    if not os.path.exists(GBENCH):
        sys.exit("build/gbench not found: run `make` first")

    all_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for spec in args.graphs:
            t0 = time.time()
            if spec.startswith("synth:"):
                path = os.path.join(tmp, "g.bin")
                run([GBENCH, "--graph", spec, "--save-graph", path])
            else:
                path = os.path.join(ROOT, spec) if not os.path.isabs(spec) else spec
            n, flags, row_ptr, col_idx = load_bin(path)
            g = to_networkx_undirected(n, row_ptr, col_idx)
            print(f"{spec}: n {n}, stored m {len(col_idx)}, undirected edges {g.number_of_edges()}, "
                  f"directed={bool(flags & 1)} (networkx build {time.time() - t0:.1f} s)")
            all_rows += check_bfs(spec, n, row_ptr, col_idx, g, pick_sources(n, row_ptr, args.sources), tmp)
            del g
            all_rows += check_pr(spec, n, row_ptr, col_idx, tmp)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
    failed = sum(r["status"] != "OK" for r in all_rows)
    print(f"\n{len(all_rows) - failed}/{len(all_rows)} checks OK; peak RSS {rss_mb:.0f} MB; wrote {args.csv}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
