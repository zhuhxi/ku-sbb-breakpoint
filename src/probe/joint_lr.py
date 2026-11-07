"""One joint break test written from the generator rules (probe/calib_break.py): the changes (level step, spread,
AR coefficient) switch on independently and can co-occur, while a no-break series may carry a whole-series trend and a
slow drift that starts at the boundary (distractor). Gaussian regression on t >= 2:
  null : x_t = c + b t/n + g ramp_t + p1 x_{t-1} + p2 x_{t-2} + e,  one noise sd         (ramp_t = (t-n0)+ / (n-n0))
  alt  : null + d step_t + q step_t x_{t-1},                       noise sd per side     (one WLS pass)
g_jlr = loglik(alt) - loglik(null); g_jlr0 = the same with no trend / ramp in either model; g_jlr_drift = g_jlr0 - g_jlr.

    python probe/joint_lr.py     # all train + test -> outputs/factory/gjlr.parquet
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def ll(e, side):
    out = 0.0
    for s in (side, ~side):
        out -= 0.5 * s.sum() * np.log(e[s].var() + 1e-12)
    return out


def lr(v, n0, drift):
    n = len(v)
    t = np.arange(2, n)
    y = v[2:]
    one, step = np.ones(n - 2), (t >= n0).astype(float)
    base = [one, v[1:-1], v[:-2]] + ([t / n, np.maximum(t - n0, 0) / (n - n0)] if drift else [])
    A0 = np.column_stack(base)
    e0 = y - A0 @ np.linalg.lstsq(A0, y, rcond=None)[0]
    l0 = -0.5 * len(y) * np.log(e0.var() + 1e-12)
    A1 = np.column_stack(base + [step, step * v[1:-1]])
    e1 = y - A1 @ np.linalg.lstsq(A1, y, rcond=None)[0]
    side = step == 1
    w = np.where(side, 1 / (e1[side].std() + 1e-9), 1 / (e1[~side].std() + 1e-9))
    e1 = y - A1 @ np.linalg.lstsq(A1 * w[:, None], y * w, rcond=None)[0]
    return ll(e1, side) - l0


def one(v, n0):
    a, b = lr(v, n0, True), lr(v, n0, False)
    return {"g_jlr": a, "g_jlr0": b, "g_jlr_drift": b - a}


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    rows = Parallel(n_jobs=30, batch_size=64)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    G.to_parquet("outputs/factory/gjlr.parquet")
    print(f"{len(G)} series in {time.time() - t0:.1f}s\n{G.describe().round(3)}")
