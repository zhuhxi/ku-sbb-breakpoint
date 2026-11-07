"""Which kind of dependence change do real breaks make? Each part standardized on its own (level / spread removed),
then |after - before| of: autocorrelation at lags 1-5 and 10, lag-2 partial autocorrelation, autocorrelation of
squares and of absolute values (volatility clustering), of signs, and spectral flatness. Label AUC per fitted group.

    python probe/dep_scan.py
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

t0 = time.time()
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
gs = pd.read_parquet("outputs/factory/genA.parquet")["g_struct"]
GROUP = {0: "stable", 1: "stable", 2: "stable", 3: "stable", 4: "ARMA11/LL", 5: "RW", 6: "SIN", 7: "SIN"}
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
N0 = S[S.period == 0].groupby("id").size()


def acf(v, k):
    return np.mean(v[k:] * v[:-k])


def stats(v):
    v = (v - v.mean()) / v.std()
    r = {f"acf{k}": acf(v, k) for k in (1, 2, 3, 4, 5, 10)}
    r["pacf2"] = (r["acf2"] - r["acf1"] ** 2) / (1 - r["acf1"] ** 2)
    q = v ** 2 - 1
    r["sq_acf1-5"] = np.mean([acf(q, k) for k in range(1, 6)]) / q.var()
    a = np.abs(v) - np.abs(v).mean()
    r["abs_acf1-5"] = np.mean([acf(a, k) for k in range(1, 6)]) / a.var()
    s = np.sign(v)
    r["sign_acf1"] = acf(s, 1)
    p = np.abs(np.fft.rfft(v))[1:] ** 2 + 1e-12
    r["spec_flat"] = np.log(np.exp(np.mean(np.log(p))) / p.mean())
    return r


def one(i, x):
    a, b = stats(x[:N0[i]]), stats(x[N0[i]:])
    return dict(id=i, **{k: abs(b[k] - a[k]) for k in a})


D = pd.DataFrame(Parallel(n_jobs=-1, batch_size=128)(delayed(one)(i, x) for i, x in zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut)))).set_index("id")
D["y"], D["grp"] = y, gs.reindex(D.index).map(GROUP)
rows = {f"{g} (n={len(E)})": {c: roc_auc_score(E.y, E[c]) for c in D.columns if c not in ("y", "grp")} for g, E in D.groupby("grp")}
print(pd.DataFrame(rows).T.round(3).to_string(), f"\n{time.time() - t0:.1f}s")
