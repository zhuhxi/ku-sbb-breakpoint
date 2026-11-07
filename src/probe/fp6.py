"""Fingerprint 6: is the level jump added to the output (mean steps up at once) or to the AR intercept
(mean climbs to the new level over ~1/(1-phi) steps)? Persistent stationary series (pre-part lag-1 corr .5-.95)
with a clear level change (|mean after - mean before| > 1 pre sd). Profile: (window mean - pre mean) / (after mean - pre mean)
for windows right after the boundary; output-add -> ~1 everywhere, intercept -> starts low and climbs.
Fakes: our no-break series plus a 2-sd jump, added to the output or through the intercept.

    python probe/fp6.py
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
W = [(0, 3), (3, 8), (8, 20), (20, 50), (50, None)]


def prof(x, n0):
    a, b = x[:n0], x[n0:]
    ac = np.corrcoef(a[1:], a[:-1])[0, 1]
    d = b.mean() - a.mean()
    if not (.5 < ac < .95 and abs(d) > a.std()):
        return None
    return [(b[s:e].mean() - a.mean()) / d for s, e in W] + [ac]


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
    return [("fake output-add", prof(out, n0)), ("fake intercept", prof(icp, n0))]


S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
N0 = S[S.period == 0].groupby("id").size()
rows = [(f"REAL label {y[i]}", prof(x, N0[i])) for i, x in zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut))]
rows += [r for o in Parallel(n_jobs=-1, batch_size=32)(delayed(fake)(s) for s in range(6000)) for r in o]
D = pd.DataFrame([[k] + p for k, p in rows if p], columns=["kind"] + [f"{s}-{e or 'end'}" for s, e in W] + ["ac"])
print("median of (window mean - pre mean) / (after mean - pre mean); 1 = jump fully there")
T = D.groupby("kind").median().round(2)
T["n"] = D.groupby("kind").size()
print(T.to_string(), f"\n{time.time() - t0:.1f}s")
