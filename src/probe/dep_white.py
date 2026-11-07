"""Dependence change after AR-whitening (ADIA 2026 forum idea, research/README.md): fit AR(5) with intercept on the
pre part, run the same filter over the after part, then look for left-over dependence in the after residuals.
  q_after   : sum of squared residual autocorrelations, lags 1-5, after part (pre residuals are white by construction)
  r1_after  : |lag-1 residual autocorrelation| after
  coef_dist : distance between AR(5) coefficients fitted before and after
  resid_var : |log(residual variance after / before)|   (spread change, for reference)
Label AUC per fitted group (genA g_struct).

    python probe/dep_white.py
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

t0 = time.time()
P = 5
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
gs = pd.read_parquet("outputs/factory/genA.parquet")["g_struct"]
GROUP = {0: "stable", 1: "stable", 2: "stable", 3: "stable", 4: "ARMA11/LL", 5: "RW", 6: "SIN", 7: "SIN"}
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
N0 = S[S.period == 0].groupby("id").size()


def lagmat(v):
    return np.column_stack([np.ones(len(v) - P)] + [v[P - k:len(v) - k] for k in range(1, P + 1)]), v[P:]


def racf(r, k):
    r = r - r.mean()
    return np.mean(r[k:] * r[:-k]) / r.var()


def one(i, x):
    n0 = N0[i]
    sc = x[:n0].std()
    a, b = x[:n0] / sc, x[n0:] / sc
    Xa, ya = lagmat(a)
    ca = np.linalg.lstsq(Xa, ya, rcond=None)[0]
    Xb, yb = lagmat(b)
    rb, ra = yb - Xb @ ca, ya - Xa @ ca
    cb_ = np.linalg.lstsq(Xb, yb, rcond=None)[0]
    return dict(id=i, q_after=sum(racf(rb, k) ** 2 for k in range(1, 6)), r1_after=abs(racf(rb, 1)),
                coef_dist=np.linalg.norm(cb_[1:] - ca[1:]), resid_var=abs(np.log(rb.var() / ra.var())))


D = pd.DataFrame(Parallel(n_jobs=-1, batch_size=128)(delayed(one)(i, x) for i, x in zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut)))).set_index("id")
D["y"], D["grp"] = y, gs.reindex(D.index).map(GROUP)
rows = {f"{g} (n={len(E)})": {c: roc_auc_score(E.y, E[c]) for c in D.columns if c not in ("y", "grp")} for g, E in D.groupby("grp")}
print(pd.DataFrame(rows).T.round(3).to_string(), "\n(raw-series dependence for comparison, probe/dep_scan.py: stable acf1 .535, SIN .601)")
print(f"{time.time() - t0:.1f}s")
