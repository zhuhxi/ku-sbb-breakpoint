"""Does the level keep moving after the boundary (a slope change), instead of a flat step? One regression per series:
    x_t = c + s t + d 1[t >= n0] + g (t - n0) 1[t >= n0] + p1 x_{t-1} + p2 x_{t-2} + e_t
slope change = g * (n - n0) / (1 - p1 - p2) / sd(e)   (extra rise over the after-part, in noise-sd units).
Stationary-pre series (ids from break_measure.parquet); real label 0 / 1 vs generated break / no-break.

    python probe/post_drift.py --menu "A switches"
"""
import argparse
import importlib.util
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ap = argparse.ArgumentParser()
ap.add_argument("--menu", default="A switches")
ap.add_argument("--n", type=int, default=3000)
args = ap.parse_args()
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


def drift(v, n0):
    n = len(v)
    t = np.arange(2, n)
    post = (t >= n0).astype(float)
    X = np.column_stack([np.ones(n - 2), t / n, post, post * (t - n0) / n, v[1:-1], v[:-2]])
    b, *_ = np.linalg.lstsq(X, v[2:], rcond=None)
    e = v[2:] - X @ b
    den = max(1 - b[4] - b[5], 0.05) * e.std()
    cov = np.linalg.pinv(X.T @ X) * e.var()
    return dict(level=b[2] / den, slope=b[3] * (n - n0) / n / den, slope_t=b[3] / np.sqrt(cov[3, 3]))


if __name__ == "__main__":
    M = pd.read_parquet("outputs/probe/break_measure.parquet")
    S = pd.read_parquet("data/train_series.parquet")
    G = {}
    for y in (0, 1):
        ids = set(M.index[M.y == y])
        G[f"real label {y}"] = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
    G[f"gen break"] = Parallel(n_jobs=30)(delayed(cb.gen)(cb.BREAKS[args.menu], 3 * 10 ** 6 + s, True, True) for s in range(args.n))
    G["gen no-break"] = Parallel(n_jobs=30)(delayed(cb.gen)(None, 3 * 10 ** 6 + s, False, True) for s in range(args.n))
    for k, L in G.items():
        D = pd.DataFrame(Parallel(n_jobs=30)(delayed(drift)(v, n0) for v, n0 in L))
        big = D[D.level.abs() > 1]
        same = np.mean(np.sign(big.slope) == np.sign(big.level))
        print(f"{k:13s} |slope_t|>2.5: {np.mean(D.slope_t.abs() > 2.5):.3f}   |slope| q50/90/99 {np.round(np.quantile(D.slope.abs(), [.5, .9, .99]), 2)}"
              f"   clear level breaks n={len(big)}: slope same direction as jump {same:.2f}, median extra rise {np.median(big.slope * np.sign(big.level)):+.2f}")
