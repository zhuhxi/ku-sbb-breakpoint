"""Fingerprint 6, one series at a time: after the boundary (pre mean removed, pre-sd units) fit
  step : 0 before position tau, constant c from tau on   (tau = 0 -> jump right at the boundary; tau > 0 -> late jump)
  ramp : rises linearly from 0 to c over L points, then flat (L small -> jump at once; L large -> slow climb)
and keep the better of the two. Same selection as probe/fp6.py; fakes with a known 2-sd jump calibrate the read-out.

    python probe/fp6b.py
"""
import importlib.util
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
B = cb.BREAKS["L abc40 ns35 shell s6"]
G = 150


def fit(x, n0):
    a, b = x[:n0], x[n0:]
    ac = np.corrcoef(a[1:], a[:-1])[0, 1]
    if not (.5 < ac < .95 and abs(b.mean() - a.mean()) > a.std()):
        return None
    b = (b - a.mean()) / a.std()
    m = len(b)
    taus = np.arange(0, min(G, m - 10))
    cs, cs2 = np.r_[0, np.cumsum(b)], np.r_[0, np.cumsum(b ** 2)]
    rest = m - taus
    sse_step = cs2[taus] + (cs2[m] - cs2[taus]) - (cs[m] - cs[taus]) ** 2 / rest
    t = np.arange(m)
    sse_ramp = []
    for L in range(1, G + 1):
        g = np.minimum((t + 1) / L, 1)
        sse_ramp.append(cs2[m] - (b @ g) ** 2 / (g @ g))
    tau, L = int(np.argmin(sse_step)), int(np.argmin(sse_ramp)) + 1
    step_better = sse_step[tau] <= min(sse_ramp)
    kind = ("jump at boundary (tau<=3)" if tau <= 3 else "late jump (tau>3)") if step_better else ("quick ramp (L<=8)" if L <= 8 else "slow climb (L>8)")
    return kind


def fake(s):
    x, n0 = cb.gen(B, 5 * 10 ** 6 + s, False, True)
    a = x[:n0]
    phi = np.corrcoef(a[1:], a[:-1])[0, 1]
    if not .5 < phi < .95:
        return []
    d = np.random.default_rng(s).choice([-1, 1]) * 2 * a.std()
    k = np.arange(len(x) - n0)
    out = x.copy(); out[n0:] += d
    icp = x.copy(); icp[n0:] += d * (1 - phi ** (k + 1))
    return [("fake instant jump", fit(out, n0)), ("fake intercept", fit(icp, n0))]


S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
N0 = S[S.period == 0].groupby("id").size()
rows = Parallel(n_jobs=-1, batch_size=64)(delayed(lambda i, x: (f"REAL label {y[i]}", fit(x, N0[i])))(i, x)
                                          for i, x in zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut)))
rows += [r for o in Parallel(n_jobs=-1, batch_size=32)(delayed(fake)(s) for s in range(3000)) for r in o]
D = pd.DataFrame([r for r in rows if r[1]], columns=["who", "kind"])
T = pd.crosstab(D.who, D.kind, normalize="index").round(2)
T["n"] = D.who.value_counts()
print(T.to_string(), f"\n{time.time() - t0:.1f}s")
