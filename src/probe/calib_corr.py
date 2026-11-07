"""Calibrate the correlated block of the no-break menu by matching the joint distribution of (acf1, acf2) over all
real label-0 series. WN / SIN / RW shares and the noise layer (normal / t / bimodal + GARCH) are the calibrated
ones; only the correlated block (35%) changes between candidates.

    python probe/calib_corr.py
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter


def acf12(v):
    v = v - v.mean()
    d = v @ v
    return (v[1:] @ v[:-1]) / d, (v[2:] @ v[:-2]) / d


def noise(r, n, df_range=(5, 10), garch_p=0.25):
    u = r.random()
    if u < 0.05:
        return r.choice([-1, 1], n) * r.uniform(1.0, 2.5) + r.standard_normal(n)
    if u < 0.525:
        df = r.uniform(*df_range)
        z = r.standard_t(df, n) / np.sqrt(df / (df - 2))
    else:
        z = r.standard_normal(n)
    if r.random() < garch_p:
        a, b = r.uniform(0.05, 0.15), r.uniform(0.5, 0.8)
        e, s2 = np.empty(n), 1.0
        for t in range(n):
            if t:
                s2 = (1 - a - b) + a * e[t - 1] ** 2 + b * s2
            e[t] = np.sqrt(s2) * z[t]
        return e
    return z


def ar(phi, e):
    return lfilter([1], np.r_[1, -np.asarray(phi)], e)


def corr_block(name, r, e, n):
    if name == "AR1 U[-.85,.85]":
        return ar([r.uniform(-0.85, 0.85)], e)
    if name == "AR1 U[-.9,.9]":
        return ar([r.uniform(-0.9, 0.9)], e)
    if name == "AR1 U[-.5,.9]":
        return ar([r.uniform(-0.5, 0.9)], e)
    if name == "AR1 U[-.9,.9] + obs":
        return ar([r.uniform(-0.9, 0.9)], e) + noise(r, n) * r.uniform(0.3, 1.0)
    if name == "AR1 U[-.9,.9] or MA1 U[-.9,.9]":
        return ar([r.uniform(-0.9, 0.9)], e) if r.random() < 0.5 else lfilter([1, r.uniform(-0.9, 0.9)], [1], e)
    if name == "AR1 or AR2 (stationary U)":
        if r.random() < 0.5:
            return ar([r.uniform(-0.9, 0.9)], e)
        while True:
            p1, p2 = r.uniform(-2, 2), r.uniform(-1, 1)
            if abs(p2) < 1 and p2 + p1 < 1 and p2 - p1 < 1:
                return ar([p1, p2], e)
    if name == "current menu":
        k = r.choice(["AR1+obs", "AR1", "AR2", "ARMA11", "MA1"], p=np.array([15, 5, 6, 5, 4]) / 35)
        if k == "AR1":
            return ar([r.uniform(-0.85, 0.85)], e)
        if k == "AR1+obs":
            return ar([r.uniform(-0.85, 0.85)], e) + noise(r, n) * r.uniform(0.3, 1.0)
        if k == "ARMA11":
            return lfilter([1, r.uniform(-0.9, 0.9)], [1, -r.uniform(-0.85, 0.85)], e)
        if k == "MA1":
            return lfilter([1, r.uniform(-0.9, 0.9)], [1], e)
        while True:
            p1, p2 = r.uniform(-1.5, 1.5), r.uniform(-0.8, 0.8)
            if abs(p2) < 1 and p2 + p1 < 1 and p2 - p1 < 1:
                return ar([p1, p2], e)
    raise KeyError(name)


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
        x = corr_block(name, r, e, n)
    return acf12(x)


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
    S = S[S.id.isin(set(lab.index[lab == 0]))]
    R = np.array([acf12(g.to_numpy()) for _, g in S.groupby("id")["value"]])
    b1, b2 = np.linspace(-1, 1, 21), np.linspace(-1, 1, 11)
    H = lambda A: np.histogram2d(A[:, 0], A[:, 1], [b1, b2])[0].ravel() / len(A)
    hr = H(R)
    print(f"real: {len(R)} series ({time.time() - t0:.1f}s). L1 distance of the (acf1, acf2) histogram, lower = closer")
    for name in ["current menu", "AR1 U[-.85,.85]", "AR1 U[-.9,.9]", "AR1 U[-.5,.9]", "AR1 U[-.9,.9] + obs",
                 "AR1 U[-.9,.9] or MA1 U[-.9,.9]", "AR1 or AR2 (stationary U)"]:
        A = np.array(Parallel(n_jobs=30)(delayed(gen)(name, s) for s in range(6000)))
        print(f"  {name:34s} L1 {np.abs(H(A) - hr).sum():.3f}")
    print(f"total {time.time() - t0:.1f}s")
