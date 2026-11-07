"""Targeted features for the weak break kinds (probe/oracle_dep.py, probe/oracle_rw.py):
  dependence : likelihood-ratio form of the oracle test with ESTIMATED pre-part AR(2) coefficients -- lag-1 coefficient
               free after the boundary (tg_dep_lr), also scaled by the pre residual variance and per post point
  random walk: on the steps (first differences) -- the single step at the boundary in pre-step sd units (tg_rw_jump),
               change in step lag-1 autocorrelation (tg_rw_ac), |log step kurtosis ratio| (tg_rw_kurt), and the
               lag-1 autocorrelation of the series itself as the "how random-walk-like" gate (tg_ac1)

    python probe/tgt_feats.py     # train + test -> outputs/factory/tgt.parquet
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()


def one(i, x, n0):
    sc = x[:n0].std() + 1e-12
    v = (x - x[:n0].mean()) / sc
    X = np.column_stack([np.ones(n0 - 2), v[1:n0 - 1], v[:n0 - 2]])
    c, p1, p2 = np.linalg.lstsq(X, v[2:n0], rcond=None)[0]
    s2 = np.mean((v[2:n0] - X @ np.array([c, p1, p2])) ** 2) + 1e-12
    t = np.arange(n0, len(v))
    yy, z = v[t] - c - p2 * v[t - 2], v[t - 1]
    r0 = yy - p1 * z
    gain = (z @ r0) ** 2 / (z @ z + 1e-12)
    st = np.diff(v)
    a, b = st[:n0 - 1], st[n0:]
    k = lambda u: np.mean(((u - u.mean()) / (u.std() + 1e-12)) ** 4)
    ac = lambda u: np.corrcoef(u[1:], u[:-1])[0, 1]
    return dict(id=i, tg_dep_lr=gain / s2, tg_dep_lr_pp=gain / s2 / len(t), tg_dep_dphi=(z @ r0) / (z @ z + 1e-12),
                tg_rw_jump=abs(st[n0 - 1] - a.mean()) / (a.std() + 1e-12), tg_rw_ac=ac(b) - ac(a),
                tg_rw_kurt=np.log(k(b) / k(a)), tg_ac1=ac(v[:n0]))


S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
N0 = S[S.period == 0].groupby("id").size()
G = pd.DataFrame(Parallel(n_jobs=30, batch_size=128)(delayed(one)(i, x, N0[i]) for i, x in zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut)))).set_index("id")
G.index = G.index.astype(str)
G.to_parquet("outputs/factory/tgt.parquet")
print(G.shape, f"{time.time() - t0:.1f}s")
