"""Per-kind Gaussian LR tests (the generator mostly changes ONE thing per break, probe/calib_break.py one_kind).
null : x_t = c + b t/n + g ramp_t + p1..p3 x_{t-1..t-3} + e, one noise sd
k_lvl : null + d step_t
k_spr : null, noise sd per side (WLS)
k_dep : null + step_t * x_{t-1..t-3}            (full AR change, gjlr only had lag 1)
k_all : all three
k_max = max of the three single kinds; k_dep_x = k_dep - max(k_lvl, k_spr) (dependence-specific evidence).

    python probe/kind_lr.py [N]   # N > 0: quick check on the first N train series (AUCs); else all -> kind_lr.parquet
"""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

P = 3


def fit(A, y, side, per_side):
    e = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    if not per_side:
        return -0.5 * len(y) * np.log(e.var() + 1e-12)
    w = np.where(side, 1 / (e[side].std() + 1e-9), 1 / (e[~side].std() + 1e-9))
    e = y - A @ np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    return sum(-0.5 * s.sum() * np.log(e[s].var() + 1e-12) for s in (side, ~side))


def one(v, n0):
    n = len(v)
    t = np.arange(P, n)
    y = v[P:]
    lags = [v[P - k:n - k] for k in range(1, P + 1)]
    step = (t >= n0).astype(float)
    side = step == 1
    base = [np.ones(len(y)), t / n, np.maximum(t - n0, 0) / (n - n0)] + lags
    A0 = np.column_stack(base)
    Al = np.column_stack(base + [step])
    Ad = np.column_stack(base + [step * l for l in lags])
    Aa = np.column_stack(base + [step] + [step * l for l in lags])
    l0 = fit(A0, y, side, False)
    r = {"k_lvl": fit(Al, y, side, False) - l0, "k_spr": fit(A0, y, side, True) - l0,
         "k_dep": fit(Ad, y, side, False) - l0, "k_all": fit(Aa, y, side, True) - l0}
    r["k_max"] = max(r["k_lvl"], r["k_spr"], r["k_dep"])
    r["k_dep_x"] = r["k_dep"] - max(r["k_lvl"], r["k_spr"])
    return r


if __name__ == "__main__":
    t0 = time.time()
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    parts = ("train",) if N else ("train", "test")
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in parts]).sort_values(["id", "time"])
    items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    if N:
        items = items[:N]
    rows = Parallel(n_jobs=30, batch_size=64)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    print(f"{len(G)} series in {time.time() - t0:.1f}s")
    if N:
        from sklearn.metrics import roc_auc_score
        y = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
        G.index = G.index.astype(str)
        J = pd.read_parquet("outputs/factory/gjlr.parquet")
        J.index = J.index.astype(str)
        G["g_jlr"] = J.loc[G.index, "g_jlr"]
        yy = y.loc[G.index].to_numpy()
        for c in G:
            print(f"  {c:8s} AUC {roc_auc_score(yy, G[c]):.4f}  corr g_jlr {np.corrcoef(G[c].rank(), G.g_jlr.rank())[0, 1]:.3f}")
    else:
        G.to_parquet("outputs/factory/kind_lr.parquet")
