"""Which change does a break make, per fitted structure (genA g_struct, whole series)? Per group and measure:
label AUC of the scale-free before-vs-after measure (0.5 = this change never happens in breaks).

    python probe/break_by_struct.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

t0 = time.time()
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
gs = pd.read_parquet("outputs/factory/genA.parquet")["g_struct"]
GROUP = {0: "stable", 1: "stable", 2: "stable", 3: "stable", 4: "ARMA11/local level", 5: "RW", 6: "SIN", 7: "SIN"}


def ac1(x):
    return np.corrcoef(x[1:], x[:-1])[0, 1]


def peak(x):
    f = np.abs(np.fft.rfft(x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))))[1:]
    k = f.argmax()
    return (k + 1) / len(x), f[k] / len(x)


def meas(v, n0):
    a, b = v[:n0], v[n0:]
    da, db = np.diff(a), np.diff(b)
    (fa, pa), (fb, pb) = peak(a), peak(b)
    return dict(level=abs(b.mean() - a.mean()) / a.std(), spread=abs(np.log(b.std() / a.std())),
                step_spread=abs(np.log(db.std() / da.std())), drift=abs(db.mean() - da.mean()) / da.std(),
                dep=abs(ac1(b) - ac1(a)), dep_diff=abs(ac1(db) - ac1(da)), freq=abs(np.log(fb / fa)), amp=abs(np.log(pb / pa)),
                local_step=abs(v[n0:n0 + 5].mean() - v[n0 - 5:n0].mean()) / da.std())


D = pd.DataFrame({i: meas(g.value.to_numpy(), int((g.period == 0).sum())) for i, g in S.groupby("id", sort=False)}).T
D["y"], D["grp"] = y, gs.reindex(D.index).map(GROUP)
rows = {}
for grp, E in D.groupby("grp"):
    rows[f"{grp} (n={len(E)})"] = {c: roc_auc_score(E.y, E[c].astype(float).fillna(0)) for c in D.columns if c not in ("y", "grp")}
print(pd.DataFrame(rows).T.round(3).to_string())
print(f"{time.time() - t0:.1f}s")
