"""Where after the boundary is the extra one-step error of real NO-BREAK series, and are the two segments equalised?
Per lag h = 1..20 after the split: median over series of z_h^2 at the true boundary / at pseudo boundaries (AR(5) fit on
the part before), real label 0 vs fake ok4 no-break. Plus |mean_a - mean_b| / sd and log(sd_b / sd_a) of the raw values.

    python probe/near_profile.py [N]
"""
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
MENU = sys.argv[2] if len(sys.argv) > 2 else "ok4"
sys.argv, N = [sys.argv[0]], int(sys.argv[1]) if len(sys.argv) > 1 else 3000
import importlib.util
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
fa = load("fa", "factory.py")


def prof(x, c, p=5, h=20):
    b, e, _, _ = fa._arfit(x[:c], p)
    o = fa._onestep(b, np.r_[x[c - p:c], x[c:c + h]], p)
    return (o / (e.std() + 1e-12)) ** 2


def one(x, n0):
    n, s = len(x), max(len(x) // 10, 25)
    ps = [prof(x, c) for c in (n0 - s, n0 + s) if 65 < c < n - 25]
    a, b = x[:n0], x[n0:]
    return prof(x, n0), (np.mean(ps, 0) if ps else np.full(20, np.nan)), abs(a.mean() - b.mean()) / x.std(), np.log(b.std() / a.std())


L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
ids = set(L.loc[L.label == 0, "id"].to_numpy()[:N])
S = pd.read_parquet("data/train_series.parquet")
S = S[S["id"].astype(str).isin(ids)].sort_values(["id", "time"])
real = [(g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for _, g in S.groupby("id", sort=False)]
fake = Parallel(n_jobs=30, batch_size=32)(delayed(cb.gen)(cb.BREAKS[MENU], 10_000 + s, False, True) for s in range(N))
for nm, D in (("real", real), ("fake", fake)):
    R = Parallel(n_jobs=30, batch_size=32)(delayed(one)(x, n0) for x, n0 in D)
    at, ps = np.array([r[0] for r in R]), np.array([r[1] for r in R])
    ratio = np.nanmean(at, 0) / np.nanmean(ps, 0)
    print(f"{nm}: at/pseudo mean z^2 by lag 1..20:", " ".join(f"{v:.2f}" for v in ratio))
    print(f"      |dmean|/sd median {np.median([r[2] for r in R]):.4f}   |log sd ratio| median {np.median(np.abs([r[3] for r in R])):.4f}")
