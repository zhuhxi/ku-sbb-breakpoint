"""Fingerprints 2-3 with the persistence mix held fixed: real vs our generator vs a burn-in version of it
(first 100 points dropped = recursion warmed up before the series starts), inside lag-1 correlation bins.
 start  = mean square of the first 5 standardized points / points 30-34   (before the boundary)
 bstart = the same right after the boundary

    python probe/fp23.py [N]
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
sys.argv = sys.argv[:1]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
B = cb.BREAKS["L abc40 ns35 shell s6"]
z = lambda v: (v - v.mean()) / v.std()


def meas(x, n0, y, kind):
    a, b = z(x[:n0]), z(x[n0:])
    return dict(kind=kind, y=y, ac=np.mean(a[1:] * a[:-1]), start=np.mean(a[:5] ** 2) / np.mean(a[30:35] ** 2),
                bstart=np.mean(b[:5] ** 2) / np.mean(b[30:35] ** 2))


def one(s):
    y = s % 2
    x, n0 = cb.gen(B, 8 * 10 ** 6 + s, bool(y), True)
    out = [meas(x, n0, y, "ours")]
    if n0 > 150:
        out.append(meas(x[100:], n0 - 100, y, "burn-in"))
    return out


F = pd.DataFrame([d for o in Parallel(n_jobs=-1, batch_size=32)(delayed(one)(s) for s in range(N)) for d in o])
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
N0 = S[S.period == 0].groupby("id").size()
R = pd.DataFrame([meas(x, N0[i], y[i], "REAL") for i, x in zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut))])
D = pd.concat([R, F])
D["ac bin"] = pd.cut(D.ac, [-1, .2, .5, .8, .95, .99, 1])
for c in ("start", "bstart"):
    print(f"\n{c} (median ratio; 1 = no edge effect)   rows: lag-1 corr bin")
    T = D.groupby(["ac bin", "kind", "y"], observed=True)[c].median().unstack(["kind", "y"]).round(2)
    print(T.to_string())
print("\nseries per bin", D.groupby(["ac bin", "kind"], observed=True).size().unstack().to_dict("index"), f"\n{time.time() - t0:.1f}s")
