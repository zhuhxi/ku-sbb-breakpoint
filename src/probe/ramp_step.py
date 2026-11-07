"""Slope or step? Around the boundary (window x[n0-w : n0+w], w in WS) four least-squares shapes are fitted:
  line   a + b t                       (smooth drift through the boundary)
  hinge  a + b t + c (t - n0)+         (drift that starts / bends at the boundary: the no-break distractor)
  step   a + b t + d 1{t >= n0}        (jump at the boundary, same slope)
  both   a + b t + c (t - n0)+ + d 1{t >= n0}
Gaussian log-likelihood gains (n/2 log RSS ratios):
  rs_step_vs_line, rs_hinge_vs_line, rs_step_vs_hinge (= step better than bend), rs_both_vs_hinge (step on top of bend)
Random-walk view on the increments d = diff(x): a step is ONE big increment at n0, a drift is a shift of the mean increment:
  rs_inc_z   (d_n0 - mean d) / sd d over the window (one-step jump)
  rs_drift_t t-statistic of mean increment after vs before (window, n0 excluded)

    python probe/ramp_step.py     # train + test -> outputs/factory/rampstep.parquet
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

WS = (30, 60, 120)


def rss(X, y):
    r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return r @ r + 1e-12


def one(x, n0):
    n = len(x)
    out = {}
    for w in WS:
        a, b = max(n0 - w, 0), min(n0 + w, n)
        y = x[a:b]
        t = np.arange(a, b) - n0
        one_, lin, hinge, step = np.ones(len(y)), t / w, np.maximum(t, 0) / w, (t >= 0).astype(float)
        R = {"line": rss(np.column_stack([one_, lin]), y), "hinge": rss(np.column_stack([one_, lin, hinge]), y),
             "step": rss(np.column_stack([one_, lin, step]), y), "both": rss(np.column_stack([one_, lin, hinge, step]), y)}
        h = len(y) / 2
        out[f"rs{w}_step_vs_line"] = h * np.log(R["line"] / R["step"])
        out[f"rs{w}_hinge_vs_line"] = h * np.log(R["line"] / R["hinge"])
        out[f"rs{w}_step_vs_hinge"] = h * np.log(R["hinge"] / R["step"])
        out[f"rs{w}_both_vs_hinge"] = h * np.log(R["hinge"] / R["both"])
        d = np.diff(x[a:b])
        k = n0 - a - 1                                    # index of x_n0 - x_{n0-1} inside d
        rest = np.delete(d, k)
        out[f"rs{w}_inc_z"] = abs(d[k] - rest.mean()) / (rest.std() + 1e-12)
        db, da = d[:k], d[k + 1:]
        out[f"rs{w}_drift_t"] = abs(da.mean() - db.mean()) / np.sqrt(db.var() / len(db) + da.var() / len(da) + 1e-12)
    return out


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    items = [(str(k), g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    rows = Parallel(n_jobs=30, batch_size=64)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    G.to_parquet("outputs/factory/rampstep.parquet")
    print(f"{len(G)} series, {G.shape[1]} cols in {time.time() - t0:.1f}s")
