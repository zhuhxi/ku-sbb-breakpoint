"""Decoy vs real level break (LESSONS 2026-11-04): ~4% of real no-break series carry a short level bump right after the
boundary that goes away after ~12-15 points; real level breaks stay. Features compare the early level after n0 with
the late level (pre window = last 100 points before n0, in pre-sd units; AR(5) one-step errors z for the whitened view).
  dc_e     raw shift of x[n0 : n0+12]            dc_l     raw shift of x[n0+20 : ] (to the end)
  dc_keep  sign(dc_e) * dc_l / (|dc_e| + .1)     dc_gap   |dc_e| - sign(dc_e) * dc_l   (how much went back)
  dc_ze    sqrt(12) * mean z_1..12               dc_zl    sqrt(m) * mean z_20.. (m points)
  dc_bump  12 * mean(z_1..12)^2 - 12 * mean(z_20..)^2 clipped at 0 is NOT used: plain difference of the two chi2 terms
  dc_ze4   sqrt(4) * mean z_1..4 (very short bumps)

    python probe/decoy_feats.py     # train + test -> outputs/factory/decoy.parquet
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("fa", "factory.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)


def one(x, n0):
    n = len(x)
    pre = x[max(n0 - 100, 0):n0]
    m0, sd = pre.mean(), pre.std() + 1e-12
    e = (x[n0:n0 + 12].mean() - m0) / sd
    late = x[n0 + 20:] if n - n0 > 40 else x[n0 + (n - n0) // 2:]
    l = (late.mean() - m0) / sd
    s = np.sign(e)
    b, r, _, _ = fa._arfit(x[:n0], 5)
    z = fa._onestep(b, np.r_[x[n0 - 5:n0], x[n0:]], 5) / (r.std() + 1e-12)
    zl = z[20:] if len(z) > 40 else z[len(z) // 2:]
    ze, zlt = np.sqrt(12) * z[:12].mean(), np.sqrt(len(zl)) * zl.mean()
    return {"dc_e": e, "dc_l": l, "dc_keep": s * l / (abs(e) + .1), "dc_gap": abs(e) - s * l, "dc_ze": ze, "dc_zl": zlt,
            "dc_bump": ze ** 2 - (zl.mean() ** 2) * 12, "dc_ze4": 2 * z[:4].mean()}


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    items = [(str(k), g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    rows = Parallel(n_jobs=30, batch_size=64)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    G.to_parquet("outputs/factory/decoy.parquet")
    print(f"{len(G)} series in {time.time() - t0:.1f}s")
