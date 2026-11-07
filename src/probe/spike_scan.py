"""Isolated spikes: how often, where (exactly at the boundary? after it?), and do they depend on the label?
AR(5) fitted on the whole series, residuals scaled by 1.4826 * MAD of the pre-boundary residuals. Spike = |z| > T with
both neighbours |z| < 3 (isolated). Per group: share of series with a spike at n0-1..n0+1, mean spikes per 1000
points before / after the boundary, share with any spike, and a random-position control for the boundary rate.
Groups: real label 0 / 1 (all train) and generated ok15 no-break / break (3000 each).

    python probe/spike_scan.py [T]
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
T = float(sys.argv[1]) if len(sys.argv) > 1 else 6
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)
rng0 = np.random.default_rng(0)


def one(x, n0, seed):
    p = 5
    A = np.column_stack([np.ones(len(x) - p)] + [x[p - k:len(x) - k] for k in range(1, p + 1)])
    e = np.r_[np.zeros(p), x[p:] - A @ np.linalg.lstsq(A, x[p:], rcond=None)[0]]
    pe = e[p:n0]
    z = (e - np.median(pe)) / (1.4826 * np.median(np.abs(pe - np.median(pe))) + 1e-12)
    z[:p] = 0
    a = np.abs(z)
    iso = (a > T) & (np.r_[0, a[:-1]] < 3) & (np.r_[a[1:], 0] < 3)
    c = int(np.random.default_rng(seed).integers(40, len(x) - 40))
    return dict(at=iso[n0 - 1:n0 + 2].any(), at_rand=iso[c - 1:c + 2].any(), pre=iso[:n0].sum() / n0 * 1000,
                post=iso[n0:].sum() / (len(x) - n0) * 1000, anyp=iso.any(), post_first20=iso[n0 + 2:n0 + 22].any())


L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
S = S.sort_values(["id", "time"])
items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
R = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(one)(x, n0, i) for i, (k, x, n0) in enumerate(items)), index=[k for k, _, _ in items])
R["grp"] = np.where(L.loc[R.index].to_numpy() == 1, "real 1", "real 0")
F = []
for brk, nm in ((False, "fake ok15 0"), (True, "fake ok15 1")):
    xs = Parallel(n_jobs=30, batch_size=32)(delayed(cb.gen)(cb.BREAKS["ok15"], 50_000 + s, brk, True) for s in range(3000))
    d = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(one)(x, n0, s) for s, (x, n0) in enumerate(xs)))
    d["grp"] = nm
    F.append(d)
D = pd.concat([R] + F)
out = D.groupby("grp").agg(at_boundary=("at", "mean"), at_random=("at_rand", "mean"), first20_after=("post_first20", "mean"),
                           per1000_pre=("pre", "mean"), per1000_post=("post", "mean"), any_spike=("anyp", "mean"), n=("at", "size"))
print(f"isolated spikes |z| > {T}:")
print(out.round(4).to_string())
