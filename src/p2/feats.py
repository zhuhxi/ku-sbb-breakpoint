"""Re-implementation (own code) of the ADIA 2025 2nd-place pipeline (aParsecFromFuture) features on KU-SBB.
X1 (8 global stats, TabPFN input) and XS (window stats on 7 transforms, global stats on 7 transforms, 6 tests on |x|).
    python p2/feats.py  -> outputs/p2/feats.parquet
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import f, ks_2samp, levene, rankdata

STATS = ["mean", "std", "min", "max", "skew", "q25", "q50", "q75", "corr"]


def skew(v):
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return np.nan
    d = v - v.mean(); m2 = (d ** 2).mean()
    return (d ** 3).mean() / m2 ** 1.5 if m2 > 0 else np.nan


def roll(v, w, fn):
    out = np.full(len(v), np.nan)
    s = np.lib.stride_tricks.sliding_window_view(v, w)
    out[w - 1:] = s.mean(1) if fn == "mean" else s.std(1, ddof=1)
    return out


def transforms(v):
    n = len(v)
    return [v, (v - v.mean()) / v.std(ddof=1), np.cumsum(v), rankdata(v, method="dense") / n, np.abs(v),
            roll(v, 16, "mean"), roll(v, 16, "std")]


def gstats(v):
    v = v[~np.isnan(v)]
    m, sd = v.mean(), v.std(ddof=1)
    return dict(mean=m, median=np.median(v), max=v.max(), min=v.min(), std=sd, skew=skew(v), mean_norm=m / sd, median_norm=np.median(v) / sd)


def wstats(v, s):
    ok = ~np.isnan(v)
    vv = v[ok]
    if len(vv) == 0:
        return {k: np.nan for k in STATS}
    q = np.quantile(vv, [.25, .5, .75], method="nearest")
    p = ~np.isnan(v) & ~np.isnan(s)
    c = np.corrcoef(v[p], s[p])[0, 1] if p.sum() > 2 and v[p].std() > 0 and s[p].std() > 0 else np.nan
    return dict(mean=vv.mean(), std=vv.std(ddof=1) if len(vv) > 1 else np.nan, min=vv.min(), max=vv.max(), skew=skew(vv),
                q25=q[0], q50=q[1], q75=q[2], corr=c)


def one(v, n0):
    out = {}
    out1 = {f"step1{k}": x for k, x in gstats(v).items()}
    T = transforms(v)
    S = [np.r_[np.nan, t[:-1]] for t in T]
    for i in (20, 60, 120, 500, 1000):
        for j, (t, s) in enumerate(zip(T, S), 1):
            g = {0: slice(max(0, n0 - i), n0), 2: slice(max(0, n0 - 2 * i), n0), 3: slice(max(0, n0 - 3 * i), n0), 1: slice(n0, n0 + i)}
            W = {p: wstats(t[sl], s[sl]) for p, sl in g.items()}
            for p in (0, 1, 2, 3):
                for k in STATS:
                    out[f"w_{k}_{j}_{i}_{p}"] = W[p][k]
            for k in ("mean", "std", "min", "max", "q25", "q50", "q75"):
                out[f"w_{k}_{j}_{i}_1_diff"] = W[0][k] - W[1][k]
                out[f"w_{k}_{j}_{i}_2_diff"] = W[2][k] - W[1][k]
                out[f"w_{k}_{j}_{i}_3_diff"] = W[3][k] - W[1][k]
            for p in (0, 2, 3):
                out[f"w_corr_{j}_{i}_{p}_diff"] = W[p]["corr"] - W[1]["corr"]
    for j, t in enumerate(T, 1):
        for k, x in gstats(t).items():
            out[f"g_{k}_{j}"] = x
    a, b = np.abs(v[:n0]), np.abs(v[n0:])
    fs = a.var(ddof=1) / b.var(ddof=1)
    out.update(f_statistic=fs, f_p_value=2 * min(f.cdf(fs, len(a) - 1, len(b) - 1), 1 - f.cdf(fs, len(a) - 1, len(b) - 1)))
    out["levene_statistic"], out["levene_p_value"] = levene(a, b)
    out["ks_statistic"], out["ks_p_value"] = ks_2samp(a, b)
    return out1, out


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")])
    S["id"] = S["id"].astype(str)
    S = S.sort_values(["id", "time"])
    items = [(k, g["value"].to_numpy(np.float64).astype(np.float32).astype(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    R = Parallel(n_jobs=30, batch_size=32)(delayed(one)(v, n0) for _, v, n0 in items)
    ids = [k for k, _, _ in items]
    X1 = pd.DataFrame([r[0] for r in R], index=ids).astype(float)
    XS = pd.DataFrame([r[1] for r in R], index=ids).astype(float)
    import os; os.makedirs("outputs/p2", exist_ok=True)
    X1.to_parquet("outputs/p2/x1.parquet"); XS.to_parquet("outputs/p2/xs.parquet")
    print(f"X1 {X1.shape}  XS {XS.shape}  {time.time() - t0:.0f}s")
