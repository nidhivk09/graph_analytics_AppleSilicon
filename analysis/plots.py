#!/usr/bin/env python3
"""All report figures from results/*.csv (PLAN.md Phase 10).

    .venv/bin/python analysis/plots.py            # every figure -> figures/*.png + *.svg
    .venv/bin/python analysis/plots.py 4 8        # only figures 4 and 8

Each figure function reads only CSVs written by the benchmark scripts; a
missing input skips that figure with a message instead of failing the rest.
Summary statistics follow gbench's: BFS GTEPS = harmonic mean over sources of
(component edges / median time of that source); PageRank = median ms/iteration.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("GGPU_RESULTS", os.path.join(ROOT, "results"))  # override for testing
FIG = os.path.join(ROOT, "figures")

DATASETS = ["ego-Facebook", "web-Google", "roadNet-CA", "soc-Pokec", "soc-LiveJournal1", "com-Orkut"]
SHORT = {"ego-Facebook": "Facebook", "web-Google": "web-Google", "roadNet-CA": "roadNet-CA",
         "soc-Pokec": "Pokec", "soc-LiveJournal1": "LiveJournal", "com-Orkut": "Orkut"}
BFS_VARIANTS = ["cpu_serial", "cpu_omp", "cpu_do", "gpu_vertex", "gpu_naive", "gpu_binned", "gpu_edge",
                "gpu_do", "hybrid"]
PR_VARIANTS = ["cpu_serial", "cpu_omp", "gpu_push", "gpu_pull_scalar", "gpu_pull_vector", "gpu_pull_adaptive"]
ORDERINGS = ["original", "random", "degree", "rcm"]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def csv(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        raise FileNotFoundError(name)
    return pd.read_csv(path)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote figures/{name}.png/.svg")


def bfs_summary(df):
    """Per (dataset, variant): harmonic-mean GTEPS over sources and median time."""
    df = df.copy()
    df["edges"] = df["gteps"] * df["time_ms_e2e"] * 1e6  # component edges (same for every trial)
    rows = []
    for (d, v), g in df.groupby(["dataset", "variant"]):
        per = g.groupby("source").agg(t=("time_ms_e2e", "median"), e=("edges", "first"))
        gt = per["e"] / (per["t"] * 1e6)
        gt = gt[gt > 0]
        rows.append({"dataset": d, "variant": v, "gteps": len(gt) / (1 / gt).sum(),
                     "median_ms": g["time_ms_e2e"].median(), "n": g["n"].iloc[0], "m": g["m"].iloc[0]})
    return pd.DataFrame(rows)


def pr_summary(df):
    df = df.copy()
    df["ms_it"] = df["time_ms_e2e"] / df["iters"]
    return df.groupby(["dataset", "variant"]).agg(ms_it=("ms_it", "median"), n=("n", "first"),
                                                  m=("m", "first")).reset_index()


def peak_gpu_bw():
    """Measured GPU sequential read bandwidth at the largest buffer (Phase 3)."""
    b = csv("micro_bandwidth.csv")
    r = b[(b.device == "gpu") & (b.op == "read")]
    return r[r.bytes == r.bytes.max()]["gbps"].median()


def grouped_bars(ax, table, groups, series, log=False, fmt=None):
    """table[series][group] -> bars; missing entries are left blank."""
    width = 0.8 / len(series)
    x = np.arange(len(groups))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(series):
        vals = [table.get(s, {}).get(g, np.nan) for g in groups]
        ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, label=s, color=cmap(i % 10))
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(g, g) for g in groups])
    if log:
        ax.set_yscale("log")


# --------------------------------------------------------------------------- figures

def fig1_bandwidth():
    """Bandwidth vs buffer size: sequential CPU vs GPU; random vs sequential gather."""
    b = csv("micro_bandwidth.csv")
    r = csv("micro_random.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    style = {"read": "-", "write": "--", "copy": ":"}
    for (dev, op), g in b.groupby(["device", "op"]):
        m = g.groupby("bytes")["gbps"].median()
        a1.plot(m.index / 2**20, m.values, style[op], marker="o", ms=3,
                color="C0" if dev == "gpu" else "C1", label=f"{dev.upper()} {op}")
    a1.axhline(102.4, color="gray", lw=0.8, ls="-.")
    a1.text(1.1, 104, "LPDDR5 theoretical 102.4 GB/s", fontsize=7, color="gray")
    a1.set_xscale("log", base=2)
    a1.set_xlabel("buffer size (MB)")
    a1.set_ylabel("GB/s")
    a1.set_title("Sequential bandwidth (copy counts read + write)")
    a1.legend(fontsize=7, ncol=2)
    for (dev, pat), g in r.groupby(["device", "pattern"]):
        m = g.groupby("table_bytes")["greads_per_s"].median()
        a2.plot(m.index / 2**20, m.values, "-" if pat == "seq" else "--", marker="o", ms=3,
                color="C0" if dev == "gpu" else "C1", label=f"{dev.upper()} {pat}")
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xlabel("table size (MB)")
    a2.set_ylabel("G reads / s (4-byte gathers)")
    a2.set_title("Gather table[idx[i]]: sequential vs random idx")
    a2.legend(fontsize=7)
    save(fig, "fig01_bandwidth")


def fig2_degrees():
    h = csv("degree_hist.csv")
    h = h[h.kind == h.kind.iloc[0]]  # symmetrized degree
    fig, ax = plt.subplots(figsize=(6, 4))
    for d in DATASETS:
        g = h[(h.dataset == d) & (h.bucket > 0) & (h["count"] > 0)]
        if g.empty:
            continue
        ax.plot(g["lo"].clip(lower=1), g["count"], marker="o", ms=3, label=SHORT[d])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree (log2 bucket lower bound)")
    ax.set_ylabel("vertices")
    ax.set_title("Degree distributions (symmetrized)")
    ax.legend(fontsize=8)
    save(fig, "fig02_degree_distributions")


def trace_df():
    for name in ("trace_bfs.csv", "trace_bfs_hybrid.csv"):
        try:
            return csv(name)
        except FileNotFoundError:
            pass
    raise FileNotFoundError("trace_bfs.csv")


def fig3_frontier():
    t = trace_df()
    t = t[t.ordering == "original"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, d in zip(axes, ["soc-LiveJournal1", "roadNet-CA"]):
        g = t[(t.dataset == d)]
        v = "gpu_edge" if "gpu_edge" in set(g.variant) else g.variant.iloc[0]
        g = g[g.variant == v]
        for i, src in enumerate(sorted(g.source.unique())[:4]):
            s = g[g.source == src]
            ax.plot(s['level'], s['frontier'], marker="." if d == "soc-LiveJournal1" else None, lw=1, label=f"source {src}")
        ax.set_yscale("log")
        ax.set_xlabel("BFS level")
        ax.set_ylabel("frontier size (vertices)")
        ax.set_title(f"{SHORT[d]}: {g.groupby('source')['level'].max().median() + 1:.0f} levels")
        ax.legend(fontsize=7)
    save(fig, "fig03_frontier_per_level")


def fig4_bfs_gteps():
    parts = [csv("bfs_cpu.csv"), csv("bfs_gpu.csv")]
    s = bfs_summary(pd.concat(parts))
    table = {v: dict(zip(g.dataset, g.gteps)) for v, g in s.groupby("variant")}
    series = [v for v in BFS_VARIANTS if v in table]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    grouped_bars(ax, table, DATASETS, series, log=True)
    ax.set_ylabel("GTEPS (harmonic mean over 64 sources)")
    ax.set_title("BFS throughput: all variants x datasets")
    ax.legend(fontsize=7, ncol=5)
    save(fig, "fig04_bfs_gteps")


def fig5_roadnet_levels():
    t = trace_df()
    t = t[(t.dataset == "roadNet-CA") & (t.ordering == "original")]
    src = sorted(t.source.unique())[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for v, c in (("gpu_do", "C3"), ("hybrid", "C2")):
        g = t[(t.variant == v) & (t.source == src)]
        if v == "hybrid" and g.params.str.contains("T=100000").any():
            g = g[g.params.str.contains("T=100000")]  # default threshold only
        if g.empty:
            continue
        a1.plot(g['level'], g['time_ms'] * 1e3, lw=0.7, color=c, label=f"{v} (device: {''.join(sorted(set(g.device)))})")
        a2.plot(g['level'], g['time_ms'].cumsum(), color=c, label=f"{v}: {g.time_ms.sum():.1f} ms total")
    a1.set_yscale("log")
    a1.set_xlabel("BFS level")
    a1.set_ylabel("time per level (us)")
    a1.set_title(f"roadNet-CA, source {src}: per-level time")
    a1.legend(fontsize=7)
    a2.set_xlabel("BFS level")
    a2.set_ylabel("cumulative ms")
    a2.set_title("Cumulative time (gpu_do: a GPU sync every level)")
    a2.legend(fontsize=7)
    save(fig, "fig05_roadnet_gpu_do_vs_hybrid")


def fig6_hybrid_sweep():
    s = csv("sweep_hybrid.csv")
    s["T"] = s.params.str.extract(r"T=(\S+)")[0]
    s["threads"] = s.params.str.extract(r"threads=(\d+)")[0].astype(int)
    pos = {"0": 10.0, "inf": 3e8}
    s["x"] = s["T"].map(lambda t: pos.get(t, None) or float(t))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, d in enumerate([d for d in DATASETS if d in set(s.dataset)]):
        for th, ls in ((4, "-"), (8, "--")):
            g = s[(s.dataset == d) & (s.threads == th)].groupby("x").time_ms_e2e.median()
            if g.empty:
                continue
            base = g.iloc[0]  # T = 0 (all GPU)
            ax.plot(g.index, g.values / base, ls, marker="o", ms=3, color=f"C{i}",
                    label=f"{SHORT[d]}, {th} CPU threads")
    ax.axvline(1e5, color="gray", lw=0.8, ls=":")
    ax.text(1.15e5, 1.6, "default T = 1e5", fontsize=7, color="gray")
    ax.set_xscale("log")
    ax.set_xticks([10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 3e8])
    ax.set_xticklabels(["0\n(all GPU)", "1e2", "1e3", "1e4", "1e5", "1e6", "1e7", "inf\n(all CPU)"])
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("threshold T (level work in edges; below T the level runs on the CPU)")
    ax.set_ylabel("time / time at T = 0")
    ax.set_title("Hybrid BFS threshold sweep")
    ax.legend(fontsize=7)
    save(fig, "fig06_hybrid_threshold_sweep")


def fig7_scaling():
    s = csv("scaling_cpu_bfs.csv")
    s["threads"] = s.params.str.extract(r"threads=(\d+)")[0]
    s["schedule"] = s.params.str.extract(r"schedule=(\S+)")[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, d in zip(axes, ["soc-LiveJournal1", "roadNet-CA"]):
        g = s[s.dataset == d]
        serial = g[g.variant == "cpu_serial"].time_ms_e2e.median()
        for sched, c in (("static", "C0"), ("dynamic/64", "C1"), ("guided", "C2")):
            h = g[(g.variant == "cpu_omp") & (g.schedule == sched)]
            m = h.groupby(h.threads.astype(int)).time_ms_e2e.median()
            ax.plot(m.index, m.values, marker="o", color=c, label=f"cpu_omp {sched}")
        h = g[g.variant == "cpu_do"]
        m = h.groupby(h.threads.astype(int)).time_ms_e2e.median()
        ax.plot(m.index, m.values, marker="s", color="C3", label="cpu_do")
        ax.axhline(serial, color="gray", ls=":", label="cpu_serial")
        ax.axvspan(4.5, 8.5, color="orange", alpha=0.08)
        ax.text(5, ax.get_ylim()[1] * 0.9 if ax.get_yscale() == "linear" else serial, "E-cores join",
                fontsize=7, color="darkorange")
        ax.set_xticks([1, 2, 4, 6, 8])
        ax.set_xlabel("OpenMP threads (1-4 P-cores, 5-8 + E-cores)")
        ax.set_ylabel("median ms per BFS")
        ax.set_title(SHORT[d])
        ax.legend(fontsize=7)
    save(fig, "fig07_openmp_scaling")


def fig8_pagerank():
    s = pr_summary(csv("pr.csv"))
    table = {v: dict(zip(g.dataset, g.ms_it)) for v, g in s.groupby("variant")}
    series = [v for v in PR_VARIANTS if v in table]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    grouped_bars(ax, table, DATASETS, series, log=True)
    ax.set_ylabel("ms per iteration (lower is better)")
    ax.set_title("PageRank: variants x datasets (20 iterations)")
    ax.legend(fontsize=7, ncol=3)
    save(fig, "fig08_pagerank_ms_per_iteration")


def fig9_reorder():
    loc = csv("reorder_locality.csv").set_index("key")
    pr = pr_summary(csv("reorder_pr.csv").assign(variant=lambda d: d.variant + "@" + d.ordering))
    bfs = bfs_summary(csv("reorder_bfs.csv").assign(variant=lambda d: d.variant + "@" + d.ordering))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    panels = [(axes[0], pr, "gpu_pull_adaptive", "ms_it", "PageRank gpu_pull_adaptive"),
              (axes[1], bfs, "hybrid", "median_ms", "BFS hybrid")]
    for ax, df, var, col, title in panels:
        table = {}
        for o in ORDERINGS:
            sub = df[df.variant == f"{var}@{o}"].set_index("dataset")[col]
            base = df[df.variant == f"{var}@original"].set_index("dataset")[col]
            table[o] = (base / sub).to_dict()
        grouped_bars(ax, table, DATASETS, ORDERINGS)
        ax.axhline(1, color="black", lw=0.8)
        width = 0.8 / len(ORDERINGS)
        for i, o in enumerate(ORDERINGS):
            for j, d in enumerate(DATASETS):
                key = f"{d}/{o}"
                if key in loc.index and d in table[o]:
                    ax.text(j + (i - 1.5) * width, table[o][d] + 0.03, f"{100 * loc.loc[key, 'same_block64']:.1f}",
                            ha="center", fontsize=5.5, rotation=90)
        ax.set_ylabel("speedup vs original ordering")
        ax.set_title(f"{title} (labels: % edges within one 64-vertex block)")
        ax.legend(fontsize=7)
    save(fig, "fig09_reordering")


def fig10_size():
    stats = csv("dataset_stats.csv").set_index("dataset")
    b = bfs_summary(pd.concat([csv("bfs_cpu.csv"), csv("bfs_gpu.csv")]))
    p = pr_summary(csv("pr.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    mb = {d: (4 * (stats.loc[d, "n"] + 1) + 4 * 2 * stats.loc[d, "undirected_edges"]) / 2**20 for d in DATASETS}
    for v, mk, c in (("hybrid", "o", "C2"), ("gpu_do", "s", "C3"), ("cpu_do", "^", "C1")):
        g = b[b.variant == v].set_index("dataset")
        ax.plot([mb[d] for d in DATASETS], [g.loc[d, "gteps"] for d in DATASETS], mk + "-", color=c,
                label=f"BFS {v} (GTEPS)")
    g = p[p.variant == "gpu_pull_adaptive"].set_index("dataset")
    ax.plot([mb[d] for d in DATASETS], [g.loc[d, "m"] / (g.loc[d, "ms_it"] * 1e6) for d in DATASETS], "D--",
            color="C4", label="PageRank gpu_pull_adaptive (G edges/s per iteration)")
    for d in DATASETS:
        ax.annotate(SHORT[d], (mb[d], 0.012), fontsize=6, rotation=90, ha="center")
    ax.axvline(8, color="gray", ls=":", lw=0.8)
    ax.text(8.5, 0.02, "8 MB SLC", fontsize=7, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("symmetric CSR size (MB)")
    ax.set_ylabel("G edges / s")
    ax.set_title("Throughput vs graph size")
    ax.legend(fontsize=7)
    save(fig, "fig10_throughput_vs_size")


def fig11_bandwidth_pct():
    peak = peak_gpu_bw()
    p = pr_summary(csv("pr.csv"))
    b = bfs_summary(pd.concat([csv("bfs_cpu.csv"), csv("bfs_gpu.csv")]))
    rows = {}
    for d in DATASETS:
        pd_ = p[(p.dataset == d) & p.variant.str.startswith("gpu_pull")]
        best = pd_.loc[pd_.ms_it.idxmin()]
        n, m = best.n, best.m
        rows.setdefault("PageRank best pull", {})[d] = 100 * (4 * (n + 1) + 8 * m + 4 * n) / (best.ms_it * 1e6) / peak
        bd = b[(b.dataset == d) & b.variant.isin(["gpu_do", "hybrid"])]
        bb = bd.loc[bd.median_ms.idxmin()]
        # Graph-streaming equivalent: one pass over row_ptr, col_idx (symmetric) and dist.
        rows.setdefault("BFS best (graph-pass equivalent)", {})[d] = \
            100 * (4 * (bb.n + 1) + 4 * bb.m + 4 * bb.n) / (bb.median_ms * 1e6) / peak
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped_bars(ax, rows, DATASETS, list(rows))
    ax.axhline(100, color="black", lw=0.8)
    ax.set_ylabel(f"% of measured peak ({peak:.0f} GB/s GPU sequential read)")
    ax.set_title("Achieved effective bandwidth")
    ax.legend(fontsize=7)
    save(fig, "fig11_bandwidth_pct_of_peak")


FIGURES = {1: fig1_bandwidth, 2: fig2_degrees, 3: fig3_frontier, 4: fig4_bfs_gteps, 5: fig5_roadnet_levels,
           6: fig6_hybrid_sweep, 7: fig7_scaling, 8: fig8_pagerank, 9: fig9_reorder, 10: fig10_size,
           11: fig11_bandwidth_pct}


def main():
    which = [int(a) for a in sys.argv[1:]] or sorted(FIGURES)
    failed = 0
    for k in which:
        print(f"figure {k}: {FIGURES[k].__doc__.strip().splitlines()[0] if FIGURES[k].__doc__ else FIGURES[k].__name__}")
        try:
            FIGURES[k]()
        except FileNotFoundError as e:
            print(f"  skipped: missing results/{e.args[0]}")
            failed += 1
        except KeyError as e:
            print(f"  skipped: no data for {e.args[0]} (has run_all.sh produced every CSV?)")
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
