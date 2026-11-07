"""Is the boundary itself special in NO-BREAK series? near20 (AR(p) fit on the part before a split, one-step errors of
the next 20 points, mean z^2; factory.fe) at the true boundary n0 vs at pseudo boundaries n0 +- shift in the SAME
series, for real label-0 and generated no-break (ok4) series. Real: boundary > pseudo => something happens at n0.

    python probe/near_self.py [N] [P]
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
P = int(sys.argv[2]) if len(sys.argv) > 2 else 20
sys.argv = [sys.argv[0]]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
fa = load("fa", "factory.py")


def near(x, c, p=P, h=20):
    b, e, _, _ = fa._arfit(x[:c], p)
    o = fa._onestep(b, np.r_[x[c - p:c], x[c:c + h]], p)
    return np.mean((o / (e.std() + 1e-12)) ** 2)


def one(x, n0):
    n = len(x)
    s = max(n // 10, 25)
    r = {"at": near(x, n0)}
    for k, c in (("m", n0 - s), ("p", n0 + s)):
        r[k] = near(x, c) if 60 + P < c < n - 25 else np.nan
    return r


L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
ids = set(L.loc[L.label == 0, "id"].to_numpy()[:N])
S = pd.read_parquet("data/train_series.parquet")
S = S[S["id"].astype(str).isin(ids)].sort_values(["id", "time"])
real = [(g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for _, g in S.groupby("id", sort=False)]
fake = Parallel(n_jobs=30, batch_size=32)(delayed(cb.gen)(cb.BREAKS["ok4"], 10_000 + s, False, True) for s in range(N))
for nm, D in (("real label 0", real), ("fake no-break", fake)):
    R = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(one)(x, n0) for x, n0 in D))
    ps = R[["m", "p"]].mean(1)
    print(f"{nm:14s} median near20: at boundary {R['at'].median():.3f}  pseudo -{R['m'].median():.3f} +{R['p'].median():.3f}  "
          f"share at > pseudo mean {(R['at'] > ps).mean():.3f}  mean log(at/pseudo) {np.log(R["at"] / ps).mean():+.3f} (se {np.log(R["at"] / ps).std() / np.sqrt(R["at"].size):.3f})")
