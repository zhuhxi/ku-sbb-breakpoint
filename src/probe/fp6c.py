"""Late jumps: where does the single best mean step sit relative to the given boundary? Whole series, step position
searched within +-150 of the boundary (offset < 0 = before it). Same selection as probe/fp6.py, plus all label-1 series.

    python probe/fp6c.py
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
BINS = [-151, -51, -21, -4, 3, 20, 50, 150]


def off(x, n0, need=True):
    a, b = x[:n0], x[n0:]
    ac = np.corrcoef(a[1:], a[:-1])[0, 1]
    if need and not (.5 < ac < .95 and abs(b.mean() - a.mean()) > a.std()):
        return None
    m = len(x)
    cs = np.r_[0, np.cumsum(x)]
    k = np.arange(max(10, n0 - 150), min(m - 10, n0 + 151))
    gain = (cs[k] / k - (cs[m] - cs[k]) / (m - k)) ** 2 * k * (m - k)   # between-segment sum of squares
    return int(k[np.argmax(gain)] - n0)


def fake(s):
    x, n0 = cb.gen(B, 5 * 10 ** 6 + s, False, True)
    a = x[:n0]
    if not .5 < np.corrcoef(a[1:], a[:-1])[0, 1] < .95:
        return None
    x = x.copy(); x[n0:] += np.random.default_rng(s).choice([-1, 1]) * 2 * a.std()
    return ("fake instant jump", off(x, n0))


S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
N0 = S[S.period == 0].groupby("id").size()
rows = Parallel(n_jobs=-1, batch_size=64)(delayed(lambda i, x: (f"REAL label {y[i]}", off(x, N0[i])))(i, x)
                                          for i, x in zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut)))
rows += [r for r in Parallel(n_jobs=-1, batch_size=32)(delayed(fake)(s) for s in range(3000)) if r]
D = pd.DataFrame([r for r in rows if r[1] is not None], columns=["who", "offset"])
D["where"] = pd.cut(D.offset, BINS)
T = pd.crosstab(D.who, D["where"], normalize="index").round(2)
T["n"] = D.who.value_counts()
print("best single mean step, offset from the given boundary (points)")
print(T.to_string())
L = D[(D.who == "REAL label 1") & (D.offset > 3)].offset
print("REAL label 1 late offsets: quartiles", L.quantile([.25, .5, .75]).tolist(), " most common", L.value_counts().head(8).to_dict())
print(f"{time.time() - t0:.1f}s")
