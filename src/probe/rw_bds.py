"""Random-walk / nonlinear-dependence tests, each on the pre and post side, then post - pre (and the pooled statistic).
Written the textbook way:
  vr_q   Lo-MacKinlay variance ratio VR(q) = var(q-sums) / (q var(1-diffs)) on the levels' increments, q = 2, 4, 8, 16
  hurst  DFA-1 exponent (scales 8..n/4, log-spaced)
  adf    ADF t-stat, constant, fixed lag 2 (statsmodels autolag is too slow); kpss level stat, nlags "auto"
  bds    BDS z-stat (Brock-Dechert-Scheinkman) on AR(2)-whitened residuals, m = 2, 3, eps = 1 sd (as Brock et al. advise,
         after the linear filter); Alphabot used min p over m on the pre side
Output columns rb_<stat>_{a,b,d}: a = pre, b = post, d = b - a.

    python probe/rw_bds.py N     # N > 0: speed + alone-AUC check on the first N train series; else all -> rw_bds.parquet
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


def vr(x, q):
    d = np.diff(x)
    if len(d) < 4 * q:
        return np.nan
    s = np.convolve(d, np.ones(q), "valid")
    return s.var() / (q * d.var() + 1e-12)


def dfa(x):
    y = np.cumsum(x - x.mean())
    n = len(y)
    sc = np.unique(np.logspace(np.log10(8), np.log10(max(n // 4, 9)), 8).astype(int))
    F = []
    for s in sc:
        k = n // s
        seg = y[:k * s].reshape(k, s)
        t = np.arange(s)
        A = np.column_stack([np.ones(s), t])
        c = np.linalg.lstsq(A, seg.T, rcond=None)[0]
        F.append(np.sqrt(((seg.T - A @ c) ** 2).mean()))
    return np.polyfit(np.log(sc), np.log(np.array(F) + 1e-12), 1)[0]


def adf(x, p=2):
    d = np.diff(x)
    y = d[p:]
    A = np.column_stack([np.ones(len(y)), x[p:-1]] + [d[p - k:-k] for k in range(1, p + 1)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ c
    s2 = e @ e / (len(y) - A.shape[1])
    return c[1] / np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1])


def kpss(x):
    e = x - x.mean()
    n = len(e)
    S = np.cumsum(e)
    L = int(np.ceil(4 * (n / 100) ** .25))
    lr = e @ e / n + 2 * sum((1 - k / (L + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, L + 1))
    return (S @ S) / (n * n * lr + 1e-12)


def bds(x, m):
    A = np.column_stack([np.ones(len(x) - 2), x[1:-1], x[:-2]])
    e = x[2:] - A @ np.linalg.lstsq(A, x[2:], rcond=None)[0]
    e = e[-600:]   # O(n^2): cap length
    I = np.abs(e[:, None] - e[None, :]) < e.std()   # eps = 1 sd
    n = len(e) - m + 1
    Im = np.ones((n, n), bool)
    for k in range(m):
        Im &= I[k:k + n, k:k + n]
    iu = np.triu_indices(n, 1)
    C1, Cm = I[k:k + n, k:k + n][iu].mean(), Im[iu].mean()
    K = ((I[:n, :n].sum(1) - 1) ** 2).sum() / (n * (n - 1) * (n - 2))   # standard estimate (Brock et al. 1996)
    s2 = 4 * (K ** m + 2 * sum(K ** (m - j) * C1 ** (2 * j) for j in range(1, m)) + (m - 1) ** 2 * C1 ** (2 * m)
              - m * m * K * C1 ** (2 * m - 2))
    return np.sqrt(n) * (Cm - C1 ** m) / np.sqrt(max(s2, 1e-12))


def stats(x):
    r = {f"vr{q}": vr(x, q) for q in (2, 4, 8, 16)}
    r.update(hurst=dfa(x), adf=adf(x), kpss=kpss(x), bds2=bds(x, 2), bds3=bds(x, 3))
    return r


def one(v, n0):
    a, b, f = stats(v[:n0]), stats(v[n0:]), stats(v)
    out = {}
    for k in a:
        out[f"rb_{k}_a"], out[f"rb_{k}_b"], out[f"rb_{k}_d"], out[f"rb_{k}_f"] = a[k], b[k], b[k] - a[k], f[k]
    return out


if __name__ == "__main__":
    t0 = time.time()
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    parts = ("train",) if N else ("train", "test")
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in parts]).sort_values(["id", "time"])
    items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    if N:
        items = items[:N]
    t1 = time.time()
    rows = Parallel(n_jobs=30, batch_size=16)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[str(k) for k, _, _ in items]).astype(float)
    print(f"{len(G)} series, compute {time.time() - t1:.1f}s (total {time.time() - t0:.1f}s)")
    if N:
        from sklearn.metrics import roc_auc_score
        y = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"].loc[G.index].to_numpy()
        auc = {c: roc_auc_score(y, G[c].fillna(G[c].median())) for c in G}
        print(pd.Series({c: max(a, 1 - a) for c, a in auc.items()}).sort_values(ascending=False).head(12).round(3).to_string())
    else:
        G.to_parquet("outputs/factory/rw_bds.parquet")
