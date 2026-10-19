#!/usr/bin/env python3
"""Phase 11: GAP Benchmark Suite bfs and pr on our datasets, as an external CPU baseline.

    scripts/gap.sh                 # clone + build GAP, then run this
    .venv/bin/python tools/run_gap.py [dataset ...]

Each graph is exported with build/gexport to a temporary .sg file (deleted
after use, so data/ never grows), and GAP runs on exactly the graph our
variants see:
  - BFS: the symmetrized graph, from the *same 64 sources* as our runs (read
    from results/bfs_cpu.csv), GTEPS with the same per-source denominators
    (undirected edges in the source's component). GAP has no warm-up and one
    source per invocation, so each run is `-r <source> -n 6` and the first
    trial is dropped (our protocol: warm-up, then 5 timed trials per source).
    Correctness: one extra `-v` run per source (GAP's verifier checks the BFS
    tree against its own serial BFS).
  - PageRank: the directed graph (GAP pulls over the in-CSR we write),
    fixed 20 iterations (`-i 20 -t 0`), 2 warm-up + 5 timed trials, same as
    gbench. Correctness: GAP's verifier at its default tolerance (1e-4).
    GAP's kernel is not identical to ours: Gauss-Seidel updates and no
    dangling-mass redistribution, so its ranks differ; only time per
    iteration is compared.
Both run with 8 and 4 OpenMP threads (all cores vs P-cores only).
Rows use our CSV schema, variants gap_bfs / gap_bfs_4t / gap_pr / gap_pr_4t:
results/gap_bfs.csv, results/gap_pr.csv, and time to convergence at tol 1e-6 in
results/gap_pr_converge.csv.
"""
import datetime
import os
import re
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAP = os.path.join(ROOT, "external", "gapbs")
DATASETS = ["ego-Facebook", "web-Google", "roadNet-CA", "soc-Pokec", "soc-LiveJournal1", "com-Orkut"]
THREADS = (8, 4)
HEADER = ("timestamp,git_commit,chip,dataset,n,m,ordering,algo,variant,params,source,trial,"
          "time_ms_e2e,time_ms_gpu,gteps,iters,validated,load1")


def sh(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def gap(binary, sg, args, threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    out = subprocess.run([os.path.join(GAP, binary), "-f", sg] + args, check=True, capture_output=True,
                         text=True, env=env).stdout
    return [float(t) * 1e3 for t in re.findall(r"^Trial Time:\s+([\d.]+)", out, re.M)], out


def main():
    datasets = sys.argv[1:] or DATASETS
    tmp = os.environ.get("GAP_TMP", os.path.join(ROOT, "build", "sg"))
    os.makedirs(tmp, exist_ok=True)
    commit = sh("git", "rev-parse", "--short", "HEAD") + ("+" if sh("git", "status", "--porcelain") else "")
    gap_commit = sh("git", "-C", GAP, "rev-parse", "--short", "HEAD")
    stats = pd.read_csv(os.path.join(ROOT, "results", "dataset_stats.csv")).set_index("dataset")
    ref = pd.read_csv(os.path.join(ROOT, "results", "bfs_cpu.csv"))
    ref = ref[ref.variant == "cpu_serial"]
    bfs_rows, pr_rows, conv_rows = [], [], []

    def row(d, algo, variant, params, source, trial, ms, gteps, iters, ok):
        now = datetime.datetime.now().isoformat(timespec="seconds")
        # m as in our CSVs: adjacency entries of the graph the algorithm runs on.
        m = 2 * int(stats.loc[d, "undirected_edges"]) if algo == "bfs" else int(stats.loc[d, "m"])
        return (f"{now},{commit},Apple M2,{d},{int(stats.loc[d, 'n'])},{m},original,{algo},"
                f"{variant},{params},{source},{trial},{ms:.4f},,{gteps:.6f},{iters},{ok},{os.getloadavg()[0]:.2f}")

    for d in datasets:
        directed = bool(stats.loc[d, "directed"])
        sym = os.path.join(tmp, f"{d}_sym.sg")
        sh("build/gexport", f"data/{d}.bin", sym, "--sym")

        # Same sources, in the same order, as every other BFS variant.
        r = ref[ref.dataset == d].copy()
        r["edges"] = r.gteps * r.time_ms_e2e * 1e6
        per_src = r.groupby("source", sort=False).edges.first()
        bad = 0
        for src, edges in per_src.items():
            _, out = gap("bfs", sym, ["-r", str(src), "-n", "1", "-v"], 8)
            ok = int("Verification:           PASS" in out)
            bad += 1 - ok
            for t in THREADS:
                times, _ = gap("bfs", sym, ["-r", str(src), "-n", "6"], t)
                variant = "gap_bfs" if t == 8 else f"gap_bfs_{t}t"
                for trial, ms in enumerate(times[1:]):  # times[0] is the warm-up
                    bfs_rows.append(row(d, "bfs", variant, f"threads={t} gap={gap_commit}", src, trial, ms,
                                        edges / (ms * 1e6), "", ok))
        print(f"{d:17s} bfs: {len(per_src)} sources, verification failures {bad}", flush=True)

        pr_sg = sym
        if directed:
            os.remove(sym)
            pr_sg = os.path.join(tmp, f"{d}.sg")
            sh("build/gexport", f"data/{d}.bin", pr_sg)
        _, out = gap("pr", pr_sg, ["-i", "100", "-n", "1", "-v"], 8)  # run to GAP tol 1e-4
        ok = int("Verification:           PASS" in out)
        for t in THREADS:
            times, _ = gap("pr", pr_sg, ["-i", "20", "-t", "0", "-n", "7"], t)
            variant = "gap_pr" if t == 8 else f"gap_pr_{t}t"
            for trial, ms in enumerate(times[2:]):  # 2 warm-up trials, as in gbench
                pr_rows.append(row(d, "pr", variant, f"iters=20 threads={t} gap={gap_commit}", "", trial, ms,
                                   stats.loc[d, "m"] * 20 / (ms * 1e6), 20, ok))
            med = sorted(times[2:])[len(times[2:]) // 2]
            print(f"{d:17s} pr  threads {t}: {med / 20:.3f} ms/iteration, verification "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
        # Time to convergence at tol 1e-6 (8 threads), as results/pr_converge.csv does
        # for our variants: Gauss-Seidel needs fewer iterations, so ms/iteration alone
        # understates GAP. -l prints one "<iter> <L1 change>" line per iteration.
        _, out = gap("pr", pr_sg, ["-i", "1000", "-t", "1e-6", "-l", "-n", "4"], 8)
        trials = re.split(r"^Trial Time:\s+([\d.]+)$", out, flags=re.M)
        for trial in range(1, 4):  # trial 0 is the warm-up
            log, secs = trials[2 * trial], float(trials[2 * trial + 1])
            iters = len(re.findall(r"^\s+\d+\s+[\d.e+-]+$", log, re.M))
            conv_rows.append(row(d, "pr", "gap_pr", f"tol=0.000001 cap=1000 threads=8 gap={gap_commit}", "",
                                 trial - 1, secs * 1e3, stats.loc[d, "m"] * iters / (secs * 1e9), iters, ok))
        print(f"{d:17s} pr  tol 1e-6: {iters} iterations, {secs * 1e3:.1f} ms", flush=True)
        os.remove(pr_sg)

    # Replace this run's datasets in the CSVs, keep the others.
    for name, rows in (("gap_bfs.csv", bfs_rows), ("gap_pr.csv", pr_rows), ("gap_pr_converge.csv", conv_rows)):
        path = os.path.join(ROOT, "results", name)
        keep = []
        if os.path.exists(path):
            with open(path) as f:
                keep = [ln.rstrip("\n") for ln in f.readlines()[1:] if ln.split(",")[3] not in datasets]
        with open(path, "w") as f:
            f.write("\n".join([HEADER] + keep + rows) + "\n")
        print(f"wrote results/{name}")


if __name__ == "__main__":
    main()
