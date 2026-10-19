#!/usr/bin/env python3
"""Markdown tables for docs/REPORT.md, computed from results/*.csv.

    .venv/bin/python analysis/report_tables.py > /tmp/tables.md

Same statistics as gbench and analysis/plots.py: BFS GTEPS = harmonic mean
over sources of (component edges / median time of that source); PageRank =
median ms/iteration. Efficiency = GTEPS per GB/s of measured peak bandwidth.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402

from plots import (DATASETS, ORDERINGS, PR_VARIANTS, BFS_VARIANTS, SHORT, bfs_summary, csv,  # noqa: E402
                   peak_gpu_bw, pr_summary)


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(x, nd=2):
    if pd.isna(x):
        return "–"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.{nd}f}" if x >= 0.1 else f"{x:.3f}"


def datasets():
    s = csv("dataset_stats.csv").set_index("dataset")
    rows = []
    for d in DATASETS:
        r = s.loc[d]
        i = lambda x: f"{int(x):,}"  # the row Series is float (mixed columns)
        rows.append([d, "directed" if r.directed else "undirected", i(r.n), i(r.m), i(r.undirected_edges),
                     f"{r.mean_deg:.1f}", i(r.max_deg), i(r.num_cc), i(r.approx_diameter), f"{r.file_mb:.0f}"])
    return md(["dataset", "type", "n", "m (stored)", "undirected edges", "mean deg", "max deg", "components",
               "approx. diameter", ".bin MB"], rows)


def bfs():
    s = bfs_summary(pd.concat([csv("bfs_cpu.csv"), csv("bfs_gpu.csv")]))
    t = {(r.dataset, r.variant): r for r in s.itertuples()}
    rows_ms, rows_gt = [], []
    for v in BFS_VARIANTS:
        rows_ms.append([v] + [fmt(t[(d, v)].median_ms) if (d, v) in t else "–" for d in DATASETS])
        rows_gt.append([v] + [fmt(t[(d, v)].gteps) if (d, v) in t else "–" for d in DATASETS])
    hdr = ["variant"] + [SHORT[d] for d in DATASETS]
    return ("BFS median ms per traversal (64 sources x 5 trials):\n\n" + md(hdr, rows_ms) +
            "\n\nBFS GTEPS (harmonic mean over sources):\n\n" + md(hdr, rows_gt))


def star():
    s = bfs_summary(csv("bfs_star.csv")).set_index("variant")
    return md(["variant", "median ms"], [[v, fmt(s.loc[v, "median_ms"])] for v in BFS_VARIANTS if v in s.index])


def pagerank():
    s = pr_summary(csv("pr.csv"))
    t = {(r.dataset, r.variant): r.ms_it for r in s.itertuples()}
    rows = [[v] + [fmt(t.get((d, v), float("nan"))) for d in DATASETS] for v in PR_VARIANTS]
    c = csv("pr_converge.csv")
    it = c.groupby(["dataset", "variant"]).iters.median()
    conv = [["iterations to 1e-6"] + [
        "/".join(sorted({str(int(it[(d, v)])) for v in PR_VARIANTS if (d, v) in it})) for d in DATASETS]]
    return md(["variant"] + [SHORT[d] for d in DATASETS], rows + conv)


def reorder():
    loc = csv("reorder_locality.csv").set_index("key")
    pr = pr_summary(csv("reorder_pr.csv").assign(variant=lambda d: d.variant + "@" + d.ordering))
    bf = bfs_summary(csv("reorder_bfs.csv").assign(variant=lambda d: d.variant + "@" + d.ordering))
    out = []
    for title, df, var, col in (("PageRank gpu_pull_adaptive ms/iteration", pr, "gpu_pull_adaptive", "ms_it"),
                                ("PageRank gpu_pull_vector ms/iteration", pr, "gpu_pull_vector", "ms_it"),
                                ("BFS hybrid median ms", bf, "hybrid", "median_ms")):
        rows = []
        for d in DATASETS:
            vals = {o: df[(df.variant == f"{var}@{o}") & (df.dataset == d)][col].iloc[0] for o in ORDERINGS}
            best = min(ORDERINGS, key=lambda o: vals[o])
            rows.append([SHORT[d]] + [fmt(vals[o]) for o in ORDERINGS] +
                        [f"{best} {vals['original'] / vals[best]:.2f}x", f"{vals['random'] / vals[best]:.2f}x"])
        out.append(f"{title}:\n\n" + md(["dataset"] + ORDERINGS + ["best vs original", "best vs random"], rows))
    rows = [[SHORT[d]] + [f"{loc.loc[f'{d}/{o}', 'mean_gap']:,.0f} / {100 * loc.loc[f'{d}/{o}', 'same_block64']:.2f}%"
                          for o in ORDERINGS] for d in DATASETS]
    out.append("Locality (mean |u-v| / % of edges within one 64-vertex block):\n\n" +
               md(["dataset"] + ORDERINGS, rows))
    return "\n\n".join(out)


def efficiency():
    peak = peak_gpu_bw()
    s = bfs_summary(pd.concat([csv("bfs_cpu.csv"), csv("bfs_gpu.csv")]))
    rows = []
    for d in DATASETS:
        g = s[s.dataset == d].set_index("variant")
        best = g.gteps.idxmax()
        rows.append([SHORT[d], best, fmt(g.loc[best, "gteps"]), f"{1000 * g.loc[best, 'gteps'] / peak:.1f}"])
    return (f"Measured peak (GPU sequential read, 2 GB buffer): {peak:.1f} GB/s.\n\n" +
            md(["dataset", "best BFS variant", "GTEPS", "MTEPS per GB/s"], rows))


SECTIONS = {"datasets": datasets, "bfs": bfs, "star": star, "pagerank": pagerank, "reorder": reorder,
            "efficiency": efficiency}

if __name__ == "__main__":
    for name in sys.argv[1:] or SECTIONS:
        print(f"<!-- {name} -->\n{SECTIONS[name]()}\n")
