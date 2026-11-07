"""Step vs ramp: three AR(2) regressions on the whole series, x_t = c + p1 x_{t-1} + p2 x_{t-2} [+ d S_t | + r R_t],
with S_t = 1[t >= n0] (a step at the boundary) and R_t = (t - n0)/(n - n0) for t >= n0 (a ramp starting there).
Features, in units of the no-change residual variance: how much the step explains, how much the ramp explains, and
step minus ramp (> 0: a step fits better). Quick CV on the 4000-series subset against the current best features.

    python probe/step_ramp.py
"""
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def step_ramp(v, n0):
    x = np.asarray(v, float)
    n = len(x)
    t = np.arange(2, n)
    base = np.column_stack([np.ones(n - 2), x[1:-1], x[:-2]])
    S = (t >= n0).astype(float)
    R = np.where(t >= n0, (t - n0) / (n - n0), 0.0)
    sse = lambda X: np.sum((x[2:] - X @ np.linalg.lstsq(X, x[2:], rcond=None)[0]) ** 2)
    s0 = sse(base)
    s2 = s0 / (n - 5) + 1e-12
    ss, sr = sse(np.column_stack([base, S])), sse(np.column_stack([base, R]))
    return {"sr_step_gain": (s0 - ss) / s2, "sr_ramp_gain": (s0 - sr) / s2, "sr_step_minus_ramp": (sr - ss) / s2}


if __name__ == "__main__":
    t0 = time.time()
    src = open("probe/feats_combo_cv.py").read().split("o0 = cv(X)")[0]   # reuse the data loading and cv()
    ns = {"__name__": "fc"}
    exec(src, ns)
    A, B, X, y, cv = ns["A"], ns["B"], ns["X"], ns["y"], ns["cv"]
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(A.index))].sort_values(["id", "time"])
    G = {i: (g["value"].to_numpy(), int((g["period"] == 0).sum())) for i, g in S.groupby("id")}
    L = pd.DataFrame(Parallel(n_jobs=12, batch_size=64)(delayed(step_ramp)(*G[i]) for i in A.index), index=A.index)
    L.to_parquet("outputs/probe/step_ramp_4000.parquet")
    print(f"features done ({time.time() - t0:.0f}s); alone AUC: " +
          "  ".join(f"{c} {max(roc_auc_score(y, L[c]), 1 - roc_auc_score(y, L[c])):.3f}" for c in L.columns))
    base = pd.concat([X, A, B], axis=1)
    o0, o1 = cv(base), cv(pd.concat([base, L], axis=1))
    boots = [np.random.default_rng(b).integers(0, len(y), len(y)) for b in range(300)]
    d = [roc_auc_score(y[i], o1[i]) - roc_auc_score(y[i], o0[i]) for i in boots]
    print(f"top-60 + old + new        AUC {roc_auc_score(y, o0):.4f}")
    print(f"  + step vs ramp          AUC {roc_auc_score(y, o1):.4f}  gain {roc_auc_score(y, o1) - roc_auc_score(y, o0):+.4f}"
          f"  95% CI {np.quantile(d, .025):+.4f} .. {np.quantile(d, .975):+.4f}")
    print(f"total {time.time() - t0:.0f}s")
