"""Official notebook clue: 'B: brief fluctuation' (one unusual reading, then back) is a deliberate non-break, and the
toy spike sits AFTER the boundary. Count outliers per series: residuals of the pre-boundary AR(2) formula with a step
dummy, robust z = e / (1.4826 MAD); spikes = |z| > 6. Split before / after the boundary, by label, real vs generated.

    python probe/spikes_count.py --n 3000
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


def count(v, n0, thr=6.0):
    n = len(v)
    X = np.column_stack([np.ones(n - 2), (np.arange(2, n) >= n0).astype(float), v[1:-1], v[:-2]])
    b, *_ = np.linalg.lstsq(X, v[2:], rcond=None)
    e = v[2:] - X @ b
    z = e / (1.4826 * np.median(np.abs(e - np.median(e))))
    big = np.abs(z) > thr
    k = n0 - 2
    i = np.flatnonzero(big)
    return dict(before=big[:k].sum() / k * 1000, after=big[k:].sum() / (len(e) - k) * 1000, any=big.any(),
                zmax=np.abs(z).max(), pos=((i - k) / len(e)).tolist())


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    S = pd.read_parquet("data/train_series.parquet")
    out = {}
    for y in (0, 1):
        ids = set(np.random.default_rng(y).choice(lab.index[lab == y], args.n, False))
        L = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
        out[f"real label {y}"] = L
    out["gen no-break"] = Parallel(n_jobs=30)(delayed(cb.gen)(None, 2 * 10 ** 6 + s, False, True) for s in range(args.n))
    for name, L in out.items():
        D = pd.DataFrame(Parallel(n_jobs=30)(delayed(count)(v, n0) for v, n0 in L))
        print(f"{name:14s} series with a |z|>6 point {D['any'].mean():.3f}   per-1000 rate before {D.before.mean():.2f} after {D.after.mean():.2f}"
              f"   max|z| q50/90/99 {np.round(np.quantile(D.zmax, [.5, .9, .99]), 1)}")
    print(f"total {time.time() - t0:.1f}s")
