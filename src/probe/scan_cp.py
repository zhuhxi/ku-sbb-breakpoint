"""Change-point scan over every split tau (Alphabot f03 / f13 idea): is the given boundary n0 the most special place?
Per kind, a split statistic for every tau in [m, n - m] via cumulative sums (O(n)):
  lvl  |mean_a - mean_b| / sd / sqrt(1/tau + 1/(n-tau))            on raw x
  spr  Gaussian LR of a variance change on AR(2) residuals (full-series fit)
  dep  |r1_a - r1_b| / sqrt(1/tau + 1/(n-tau)), r1 = lag-1 autocorrelation of raw x
Features per kind: sc_<k>_rel = T(n0) / max T, sc_<k>_dist = |argmax - n0| / n, sc_<k>_rank = share of tau with T < T(n0).

    python probe/scan_cp.py     # train + test -> outputs/factory/scan_cp.parquet
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

M = 20


def one(v, n0):
    n = len(v)
    tau = np.arange(M, n - M)
    a, b = tau, n - tau
    c1, c2 = np.cumsum(v), np.cumsum(v * v)
    S1, S2 = c1[tau - 1], c2[tau - 1]
    sd = np.sqrt(max(v.var(), 1e-12))
    T = {"lvl": np.abs(S1 / a - (c1[-1] - S1) / b) / sd / np.sqrt(1 / a + 1 / b)}
    A = np.column_stack([np.ones(n - 2), v[1:-1], v[:-2]])
    e = np.r_[0, 0, v[2:] - A @ np.linalg.lstsq(A, v[2:], rcond=None)[0]]
    q = np.cumsum(e * e)[tau - 1]
    Q = np.sum(e * e)
    T["spr"] = n * np.log(Q / n + 1e-12) - a * np.log(q / a + 1e-12) - b * np.log((Q - q) / b + 1e-12)
    x = v - v.mean()
    cp = np.r_[0, np.cumsum(x[1:] * x[:-1])]          # cp[t] = sum_{s<=t} x_s x_{s-1}
    cx = np.cumsum(x * x)
    ra = cp[tau - 1] / (cx[tau - 1] + 1e-12)
    rb = (cp[-1] - cp[tau - 1]) / (cx[-1] - cx[tau - 1] + 1e-12)
    T["dep"] = np.abs(ra - rb) / np.sqrt(1 / a + 1 / b)
    i0 = int(np.clip(n0 - M, 0, len(tau) - 1))
    r = {}
    for k, t in T.items():
        r[f"sc_{k}_rel"] = t[i0] / (t.max() + 1e-12)
        r[f"sc_{k}_dist"] = abs(tau[t.argmax()] - n0) / n
        r[f"sc_{k}_rank"] = (t < t[i0]).mean()
    return r


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    items = [(str(k), g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    t1 = time.time()
    rows = Parallel(n_jobs=30, batch_size=256)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    G.to_parquet("outputs/factory/scan_cp.parquet")
    print(f"{len(G)} series, compute {time.time() - t1:.1f}s, total {time.time() - t0:.1f}s")
