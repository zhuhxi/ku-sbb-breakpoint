"""Matched-size likelihood-ratio features: the generator's break sizes are known (level 2 and 0.85 noise sd, spread
x2 or x1/2, lag-1 dependence +-0.5, noise type redrawn), so score each series by how much better "this exact break"
explains the post-boundary residuals than "no break" (Neyman-Pearson: the most powerful test against a known
alternative). Residuals come from the pre-boundary AR(2) fit. Quick CV on the 4000-series subset against
top-60 + structure-aware + break-size features (probe/feats_combo_cv.py).

    python probe/lr_feats.py
"""
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def lr_feats(v, n0):
    x = np.asarray(v, float)
    n = len(x)
    X = np.column_stack([np.ones(n - 2), x[1:-1], x[:-2]])
    b, *_ = np.linalg.lstsq(X[: n0 - 2], x[2:n0], rcond=None)
    e = x[2:] - X @ b                                         # residuals of the pre-boundary formula, whole series
    k = n0 - 2
    s0 = e[:k].std() + 1e-12
    z0, z1 = e[:k] / s0, e[k:] / s0
    f = {}
    # level: a mean shift of d noise sd shows in the residuals as d * (1 - p1 - p2)
    g = 1 - b[1] - b[2]
    for d in (2.0, 0.85):
        dl = d * g
        f[f"lr_level_{d}"] = max(np.sum(z1 ** 2 - (z1 - s * dl) ** 2) / 2 for s in (1, -1))
    # spread: post residual sd multiplied by k
    for kk, nm in ((2.0, "x2"), (0.5, "half")):
        f[f"lr_spread_{nm}"] = np.sum(-np.log(kk) - z1 ** 2 / (2 * kk ** 2) + z1 ** 2 / 2)
    # dependence: lag-1 coefficient moves by +-0.5
    xc = (x[1:-1] - x[:n0].mean())[k:] / s0
    f["lr_dep_0.5"] = max(np.sum(z1 ** 2 - (z1 - s * 0.5 * xc) ** 2) / 2 for s in (1, -1))
    # noise type: t(6) vs normal log-likelihood after the boundary minus before (per point)
    tl = lambda z: (stats.t.logpdf(z * np.sqrt(1.5), 6) + 0.5 * np.log(1.5) - stats.norm.logpdf(z)).mean()
    f["lr_tail_change"] = tl(z1 / (z1.std() + 1e-12)) - tl(z0)
    # drift vs step after the boundary (a distractor in both labels): ramp minus step fit, per point
    t = np.arange(len(z1), dtype=float)
    sse_step = np.sum((z1 - z1.mean()) ** 2)
    sse_ramp = np.sum((z1 - np.polyval(np.polyfit(t, z1, 1), t)) ** 2) if len(z1) > 2 else sse_step
    f["lr_ramp_vs_step"] = (sse_step - sse_ramp) / len(z1)
    f["lr_best"] = max(f["lr_level_2.0"], f["lr_level_0.85"], f["lr_spread_x2"], f["lr_spread_half"], f["lr_dep_0.5"])
    return f


if __name__ == "__main__":
    t0 = time.time()
    import importlib.util
    s = importlib.util.spec_from_file_location("fc", "probe/feats_combo_cv.py")
    src = open("probe/feats_combo_cv.py").read().split("o0 = cv(X)")[0]   # reuse the data loading and cv()
    ns = {"__name__": "fc"}
    exec(src, ns)
    A, B, X, y, cv = ns["A"], ns["B"], ns["X"], ns["y"], ns["cv"]
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(A.index))].sort_values(["id", "time"])
    G = {i: (g["value"].to_numpy(), int((g["period"] == 0).sum())) for i, g in S.groupby("id")}
    L = pd.DataFrame(Parallel(n_jobs=12, batch_size=64)(delayed(lr_feats)(*G[i]) for i in A.index), index=A.index)
    L.to_parquet("outputs/probe/lr_feats_4000.parquet")
    print(f"features done ({time.time() - t0:.0f}s); alone AUC: " +
          "  ".join(f"{c} {max(roc_auc_score(y, L[c]), 1 - roc_auc_score(y, L[c])):.3f}" for c in L.columns))
    base = pd.concat([X, A, B], axis=1)
    o0, o1 = cv(base), cv(pd.concat([base, L], axis=1))
    boots = [np.random.default_rng(b).integers(0, len(y), len(y)) for b in range(300)]
    d = [roc_auc_score(y[i], o1[i]) - roc_auc_score(y[i], o0[i]) for i in boots]
    print(f"top-60 + old + new        AUC {roc_auc_score(y, o0):.4f}")
    print(f"  + likelihood-ratio      AUC {roc_auc_score(y, o1):.4f}  gain {roc_auc_score(y, o1) - roc_auc_score(y, o0):+.4f}"
          f"  95% CI {np.quantile(d, .025):+.4f} .. {np.quantile(d, .975):+.4f}")
    print(f"total {time.time() - t0:.0f}s")
