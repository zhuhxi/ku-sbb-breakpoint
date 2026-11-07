"""The last ~2%: correlated-third no-break series that no simple formula matches even after removing a straight line.
Expert checklist (on the detrended series) vs series that do match, a boundary check (is something happening exactly
at the official boundary? -> possibly a break labelled 0), and a plot.

    python probe/rest2.py
"""
import importlib.util
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0], "--n", "1"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
em = load("em", "probe/exact_match.py")
ed = load("ed", "probe/expert_diag.py")


def split_stats(x, k):
    a, b = x[:k], x[k:]
    tm = abs(a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
    lv = abs(np.log(a.var() / b.var()))
    da, db = np.diff(a), np.diff(b)
    return tm, lv, abs(np.corrcoef(a[1:], a[:-1])[0, 1] - np.corrcoef(b[1:], b[:-1])[0, 1])


def one(v, n0):
    t = np.arange(len(v))
    d = v - np.polyval(np.polyfit(t, v, 1), t)
    win, ok = em.one(v)
    wd, okd = em.one(d)
    out = dict(win=win, plain=any(ok.values()), detr=any(okd.values()))
    if win.startswith("SIN") or out["plain"] or out["detr"]:
        return out
    x = (d - d.mean()) / d.std()
    out.update(ed.diag(d))
    bt = split_stats(x, n0)                                   # at the official boundary
    ks = np.linspace(0.2, 0.8, 25) * len(x)
    al = np.array([split_stats(x, int(k)) for k in ks])        # at other split points
    out.update({"B mean jump at boundary (t)": bt[0], "B mean jump rank vs other splits": (al[:, 0] < bt[0]).mean(),
                "B log var ratio at boundary": bt[1], "B var ratio rank vs other splits": (al[:, 1] < bt[1]).mean(),
                "B acf1 change at boundary": bt[2], "B acf1 change rank vs other splits": (al[:, 2] < bt[2]).mean()})
    return out


if __name__ == "__main__":
    t0 = time.time()
    R = np.load("outputs/probe/real_acf12_l0.npy")
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    ids0 = np.sort(lab.index[lab == 0].to_numpy())
    keep = set(ids0[(np.abs(R[:, 0]) >= 0.1) & (R[:, 0] <= 0.9)])
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(keep)].sort_values(["id", "time"])
    G = {k: (g["value"].to_numpy(), int((g["period"] == 0).sum())) for k, g in S.groupby("id")}
    out = dict(zip(G, Parallel(n_jobs=30)(delayed(one)(v, n0) for v, n0 in G.values())))
    D = pd.DataFrame.from_dict(out, orient="index")
    nonsin = ~D.win.str.startswith("SIN")
    rest = D[nonsin & ~D.plain & ~D.detr]
    print(f"label-0 total {len(ids0)}, correlated-filter {len(D)}, non-sine {nonsin.sum()}, "
          f"plain match {D.plain[nonsin].mean():.3f}, +trend match {(D.detr & ~D.plain)[nonsin].mean():.3f}, "
          f"left {len(rest)} = {len(rest) / len(ids0):.3%} of all label-0   ({time.time() - t0:.1f}s)")
    rest.drop(columns=["win", "plain", "detr"]).to_parquet("outputs/probe/rest2.parquet")
    cols = [c for c in rest.columns if c[0].isdigit() or c.startswith("B ")]
    print(rest[cols].astype(float).quantile([.25, .5, .75]).T.round(3).to_string())
    pick = np.random.default_rng(0).choice(rest.index, 8, replace=False)
    fig, ax = plt.subplots(8, 2, figsize=(16, 16), gridspec_kw={"width_ratios": [5, 1.5]})
    for i, k in enumerate(pick):
        v, n0 = G[k]
        t = np.arange(len(v))
        ax[i, 0].plot(v, lw=0.6, color="tab:red")
        ax[i, 0].axvline(n0, color="k", ls="--", lw=1)
        ax[i, 0].plot(t, np.polyval(np.polyfit(t, v, 1), t), color="tab:blue", lw=1)
        ax[i, 0].set_title(f"id={k}  n={len(v)}  (dashed = official boundary, blue = fitted line)", fontsize=9, loc="left")
        d = v - np.polyval(np.polyfit(t, v, 1), t)
        ax[i, 1].bar(range(1, 41), ed.pacf(d, 40)[1:], color="tab:red")
        ax[i, 1].axhline(2 / np.sqrt(len(v)), color="k", lw=0.5, ls=":")
        ax[i, 1].axhline(-2 / np.sqrt(len(v)), color="k", lw=0.5, ls=":")
        ax[i, 1].set_title("PACF after removing line", fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/probe/rest2.png", dpi=85)
    print(f"total {time.time() - t0:.1f}s")
