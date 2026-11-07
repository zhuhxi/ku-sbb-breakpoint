"""Official notebook 'B: brief fluctuation': is there a short stretch (w = 10 or 25 points) that sits high/low or wiggles
hard, then returns? Residuals of the pre-boundary AR(2) formula with a step dummy (so a lasting level change is
removed), robust z = e / (1.4826 MAD). Per segment (before / after the boundary):
  level bump  = max over windows of |mean z| * sqrt(w)
  wiggle bump = max over windows of mean z^2
Real label 0 / label 1 vs generated no-break series.

    python probe/bumps.py --n 3000
"""
import argparse
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000)
args = ap.parse_args()
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)
W = (10, 25)


def bumps(v, n0):
    n = len(v)
    X = np.column_stack([np.ones(n - 2), (np.arange(2, n) >= n0).astype(float), v[1:-1], v[:-2]])
    b, *_ = np.linalg.lstsq(X, v[2:], rcond=None)
    e = v[2:] - X @ b
    z = e / (1.4826 * np.median(np.abs(e - np.median(e))))
    k = n0 - 2
    f = {}
    for w in W:
        rm = np.convolve(z, np.ones(w) / w, "valid") * np.sqrt(w)
        rv = np.convolve(z ** 2, np.ones(w) / w, "valid")
        for side, sl in (("before", slice(0, k - w + 1)), ("after", slice(k, None))):
            f[f"lvl{w}_{side}"] = np.abs(rm[sl]).max()
            f[f"wig{w}_{side}"] = rv[sl].max()
    return f


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    S = pd.read_parquet("data/train_series.parquet")
    G = {}
    for y in (0, 1):
        ids = set(np.random.default_rng(y).choice(lab.index[lab == y], args.n, False))
        G[f"real label {y}"] = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
    G["gen no-break"] = Parallel(n_jobs=30)(delayed(cb.gen)(None, 2 * 10 ** 6 + s, False, True) for s in range(args.n))
    R = {k: pd.DataFrame(Parallel(n_jobs=30)(delayed(bumps)(v, n0) for v, n0 in L)) for k, L in G.items()}
    # 'big' = above the 95% quantile of the generated no-break series (5% expected if same)
    for c in [f"{t}{w}" for t in ("lvl", "wig") for w in W]:
        thr = {s: R["gen no-break"][f"{c}_{s}"].quantile(.95) for s in ("before", "after")}
        row = "  ".join(f"{k}: before {np.mean(D[f'{c}_before'] > thr['before']):.3f} after {np.mean(D[f'{c}_after'] > thr['after']):.3f}" for k, D in R.items())
        print(f"{c:6s} share above gen-95%  {row}")
    print(f"total {time.time() - t0:.1f}s")
