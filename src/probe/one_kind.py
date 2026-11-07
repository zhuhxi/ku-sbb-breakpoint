"""Does a break change one thing (ekhi-style: one of mean / spread / dependence) or several at once (our switches)?
Stable series only. Per measure (level, spread, dependence, shape) a series counts as changed when the measure is above
the 95th percentile of the label-0 series of the same source; then the share of label-1 series with 0 / 1 / 2+ changes.

    python probe/one_kind.py
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()
ARGS = sys.argv[1:]                                            # calib_break.py overwrites sys.argv on import
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
base = cb.BREAKS["L abc40 ns35 shell s6"]
MENUS = {"ok1 (one kind, new)": cb.BREAKS["ok1"],
         "ours (independent switches)": base,
         "one kind per break": dict(base, switches=None, p=dict(level=.4, spread=.4, dep=.2), level_unit="noise")}
if ARGS:                                                       # menus by name from the command line
    MENUS = {m: cb.BREAKS[m] for m in ARGS}
M = ["level", "spread", "dep", "shape"]


def meas(x, n0):
    a, b = x[:n0], x[n0:]
    k = lambda v: np.mean(((v - v.mean()) / v.std()) ** 4)
    ac = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]
    return dict(level=abs(b.mean() - a.mean()) / a.std(), spread=abs(np.log(b.std() / a.std())), dep=abs(ac(b) - ac(a)), shape=abs(np.log(k(b) / k(a))))


def count(D):
    q = D[D.y == 0][M].quantile(.95)
    c = (D[D.y == 1][M] > q).sum(axis=1)
    return {"0": round(float((c == 0).mean()), 2), "1": round(float((c == 1).mean()), 2), "2+": round(float((c >= 2).mean()), 2)}, \
        {m: round(float((D[D.y == 1][m] > q[m]).mean()), 2) for m in M}


def fake(B, s):
    x, n0, info = cb.gen(B, 3 * 10 ** 6 + s, bool(s % 2), "info")
    return dict(meas(x, n0), y=s % 2) if info["structure"] in ("WN", "AR1", "AR2") else None


S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
gs = pd.read_parquet("outputs/factory/genA.parquet")["g_struct"]
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
N0 = S[S.period == 0].groupby("id").size()
R = pd.DataFrame([dict(meas(x, N0[i]), y=y[i]) for i, x in zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut)) if gs.get(i, -1) in (0, 1, 2, 3)])
print(f"{'REAL stable':30s}", *count(R))
for name, B in MENUS.items():
    D = pd.DataFrame([d for d in Parallel(n_jobs=-1, batch_size=64)(delayed(fake)(B, s) for s in range(6000)) if d])
    print(f"{name:30s}", *count(D))
print(f"{time.time() - t0:.1f}s")
