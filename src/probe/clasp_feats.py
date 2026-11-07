"""ClaSP-style score at the known boundary: cut each series into sliding windows, label each window before/after,
and see how well a 3-nearest-neighbour vote (neighbours closer than one window in time excluded) recovers the side.
High AUC = the two sides look different in some way, whatever the change is. On the raw series and on AR(10)
residuals. Quick CV on the 4000-series subset against the current best features.

    python probe/clasp_feats.py
"""
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


def side_auc(x, n0, w=10, step=2, k=3):
    s = np.arange(0, len(x) - w + 1, step)
    W = np.stack([x[i:i + w] for i in s])
    W = (W - x[:n0].mean()) / (x[:n0].std() + 1e-12)
    y = (s + w // 2 >= n0).astype(float)
    d, idx = NearestNeighbors(n_neighbors=k + 2 * w // step + 1).fit(W).kneighbors(W)
    far = np.abs(s[idx] - s[:, None]) >= w                         # skip trivial matches (overlapping windows)
    vote = np.array([y[idx[i][far[i]][:k]].mean() for i in range(len(s))])
    return roc_auc_score(y, vote)


def ar_resid(v, p=10):
    X = np.column_stack([np.ones(len(v) - p)] + [v[p - j - 1:len(v) - j - 1] for j in range(p)])
    return v[p:] - X @ np.linalg.lstsq(X, v[p:], rcond=None)[0]


def clasp(v, n0):
    v = np.asarray(v, float)
    return {"clasp_raw": side_auc(v, n0), "clasp_res": side_auc(ar_resid(v), n0 - 10),
            "clasp_absres": side_auc(np.abs(ar_resid(v)), n0 - 10)}


if __name__ == "__main__":
    t0 = time.time()
    src = open("probe/feats_combo_cv.py").read().split("o0 = cv(X)")[0]
    ns = {"__name__": "fc"}
    exec(src, ns)
    A, B, X, y, cv = ns["A"], ns["B"], ns["X"], ns["y"], ns["cv"]
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(A.index))].sort_values(["id", "time"])
    G = {i: (g["value"].to_numpy(), int((g["period"] == 0).sum())) for i, g in S.groupby("id")}
    L = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(clasp)(*G[i]) for i in A.index), index=A.index)
    L.to_parquet("outputs/probe/clasp_4000.parquet")
    print(f"features done ({time.time() - t0:.0f}s); alone AUC: " +
          "  ".join(f"{c} {max(roc_auc_score(y, L[c]), 1 - roc_auc_score(y, L[c])):.3f}" for c in L.columns))
    base = pd.concat([X, A, B], axis=1)
    o0, o1 = cv(base), cv(pd.concat([base, L], axis=1))
    boots = [np.random.default_rng(b).integers(0, len(y), len(y)) for b in range(300)]
    d = [roc_auc_score(y[i], o1[i]) - roc_auc_score(y[i], o0[i]) for i in boots]
    print(f"top-60 + old + new        AUC {roc_auc_score(y, o0):.4f}")
    print(f"  + ClaSP scores          AUC {roc_auc_score(y, o1):.4f}  gain {roc_auc_score(y, o1) - roc_auc_score(y, o0):+.4f}"
          f"  95% CI {np.quantile(d, .025):+.4f} .. {np.quantile(d, .975):+.4f}")
    print(f"total {time.time() - t0:.0f}s")
