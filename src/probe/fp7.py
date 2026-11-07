"""Fingerprint 7: is the spread change applied to the output (x * k after the boundary: the spread jumps at once)
or to the innovations before the AR recursion (spread climbs over ~1/(1-phi^2) points)? Persistent stationary
series (pre lag-1 corr .5-.95), clear spread change (|log sd ratio| > log 1.4), small level change (< .5 pre sd).
Profile: log(mean square of window around the pre mean / pre var) / log(same for the whole after part); 1 = fully there.
Fakes: AR(1) with burn-in, k = 2 or 1/2, applied to the output or to the innovations.

    python probe/fp7.py
"""
import time

import numpy as np
import pandas as pd
from scipy.signal import lfilter

t0 = time.time()
W = [(0, 5), (5, 15), (15, 40), (40, 100), (100, None)]


def prof(x, n0):
    a, b = x[:n0], x[n0:]
    ac = np.corrcoef(a[1:], a[:-1])[0, 1]
    m, v = a.mean(), a.var()
    lr = np.log(np.mean((b - m) ** 2) / v)
    if not (.5 < ac < .95 and abs(np.log(b.std() / a.std())) > np.log(1.4) and abs(b.mean() - m) < .5 * a.std()):
        return None
    return [np.log(np.mean((b[s:e] - m) ** 2) / v) / lr for s, e in W]


rows = []
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
N0 = S[S.period == 0].groupby("id").size()
for i, x in zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut)):
    rows.append((f"REAL label {y[i]}", prof(x, N0[i])))
r = np.random.default_rng(0)
for s in range(6000):
    n = int(r.integers(256, 1025)); n0 = int(r.uniform(.3, .7) * n); phi = r.uniform(.5, .95)
    k = 2.0 if r.random() < .5 else .5
    e = r.standard_normal(300 + n)
    out = lfilter([1], [1, -phi], e)[300:]; out[n0:] *= k
    e2 = e.copy(); e2[300 + n0:] *= k
    inn = lfilter([1], [1, -phi], e2)[300:]
    rows += [("fake output-scale", prof(out, n0)), ("fake innovation-scale", prof(inn, n0))]
D = pd.DataFrame([[w] + p for w, p in rows if p], columns=["who"] + [f"{s}-{e or 'end'}" for s, e in W])
T = D.groupby("who").median().round(2)
T["n"] = D.who.value_counts()
print("median share of the spread change already there, per window after the boundary")
print(T.to_string(), f"\n{time.time() - t0:.1f}s")
