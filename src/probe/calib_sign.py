"""Does a positive-leaning correlated block fit better? Compare (acf1, acf2) histogram L1 and the share of series with
acf1 > 0.1 / acf1 < -0.1 between real label-0 series and the calibrated menu with different correlated blocks.

    python probe/calib_sign.py
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("cc", "probe/calib_corr.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
ar, noise = cc.ar, cc.noise
CANDS = {
    "AR1 U[-.9,.9]          (symmetric)": lambda r, e, n: ar([r.uniform(-0.9, 0.9)], e),
    "AR1 U[-.5,.9]": lambda r, e, n: ar([r.uniform(-0.5, 0.9)], e),
    "AR1 U[0,.9]            (positive only)": lambda r, e, n: ar([r.uniform(0, 0.9)], e),
    "AR1 U[-.9,.9] + obs": lambda r, e, n: ar([r.uniform(-0.9, 0.9)], e) + noise(r, n) * r.uniform(0.5, 1.5),
    "AR1 U[-.5,.9] + obs": lambda r, e, n: ar([r.uniform(-0.5, 0.9)], e) + noise(r, n) * r.uniform(0.5, 1.5),
    "AR1 U[0,.9] + obs": lambda r, e, n: ar([r.uniform(0, 0.9)], e) + noise(r, n) * r.uniform(0.5, 1.5),
}


def gen(name, seed):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    e, t = noise(r, n), np.arange(n)
    u = r.random()
    if u < 0.42:
        x = e
    elif u < 0.58:
        x = r.uniform(0.5, 3) * np.sin(2 * np.pi * t / r.integers(12, 33) + r.uniform(0, 2 * np.pi)) + e
    elif u < 0.65:
        x = np.cumsum(e)
    else:
        x = CANDS[name](r, e, n)
    return cc.acf12(x)


if __name__ == "__main__":
    t0 = time.time()
    f = "outputs/probe/real_acf12_l0.npy"
    if not os.path.exists(f):
        lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
        S = pd.read_parquet("data/train_series.parquet", columns=["id", "time", "value"]).sort_values(["id", "time"])
        S = S[S.id.isin(set(lab.index[lab == 0]))]
        np.save(f, np.array([cc.acf12(g.to_numpy()) for _, g in S.groupby("id")["value"]]))
    R = np.load(f)
    b1, b2 = np.linspace(-1, 1, 21), np.linspace(-1, 1, 11)
    H = lambda A: np.histogram2d(A[:, 0], A[:, 1], [b1, b2])[0].ravel() / len(A)
    sign = lambda A: f"pos {np.mean(A[:, 0] > 0.1):.3f}  neg {np.mean(A[:, 0] < -0.1):.3f}"
    print(f"real ({len(R)}): {sign(R)}   ({time.time() - t0:.1f}s)")
    for name in CANDS:
        A = np.array(Parallel(n_jobs=30)(delayed(gen)(name, s) for s in range(6000)))
        print(f"  {name:40s} L1 {np.abs(H(A) - H(R)).sum():.3f}   {sign(A)}")
    print(f"total {time.time() - t0:.1f}s")
