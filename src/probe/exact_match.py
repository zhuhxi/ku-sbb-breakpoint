"""Ventilator-style per-series exact matching for the correlated third of no-break series: fit each candidate formula
by MLE, call it an exact match when its residuals are pure noise (Ljung-Box p > 0.05 at lag 10). A known-AR1 control
shows what a true match looks like (about 95% pass).

    python probe/exact_match.py --n 600
"""
import argparse
import importlib.util
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=600)
args = ap.parse_args()
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("mle", "probe/mle_noise.py")
mle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mle)
MODELS = ["WN", "AR1", "AR2", "MA1", "ARMA11"]


def ljung_p(e, h=10):
    e = e - e.mean()
    r = np.array([(e[k:] @ e[:-k]) / (e @ e) for k in range(1, h + 1)])
    n = len(e)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, h + 1)))
    return stats.chi2.sf(q, h)


def one(v):
    _, win, res = mle.fit("x", v)
    return win, {m: ljung_p(res[m]["e"]) > 0.05 for m in MODELS}


if __name__ == "__main__":
    t0 = time.time()
    R = np.load("outputs/probe/real_acf12_l0.npy")
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    ids0 = np.sort(lab.index[lab == 0].to_numpy())                 # same order as the cached acf (groupby id)
    keep = ids0[(np.abs(R[:, 0]) >= 0.1) & (R[:, 0] <= 0.9)]
    ids = set(np.random.default_rng(0).choice(keep, args.n, replace=False))
    S = pd.read_parquet("data/train_series.parquet", columns=["id", "time", "value"])
    S = S[S.id.isin(ids)].sort_values(["id", "time"])
    real = [g.to_numpy() for _, g in S.groupby("id")["value"]]
    r = np.random.default_rng(1)
    ctrl = [mle.ar if False else None for _ in range(0)]
    ctrl = []
    for _ in range(200):                                           # control: true AR1, normal noise
        n, phi = int(r.integers(256, 1025)), r.uniform(-0.85, 0.85)
        e, x = r.standard_normal(n), np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e[t]
        ctrl.append(x)
    out = Parallel(n_jobs=30)(delayed(one)(v) for v in real + ctrl)
    def show(name, o):
        M = pd.DataFrame([d for _, d in o])
        anym = M.any(axis=1)
        print(f"{name} ({len(o)} series): exact-match rate per formula")
        print("   " + "  ".join(f"{m} {M[m].mean():.2f}" for m in MODELS) + f"   | any {anym.mean():.2f}  none {1 - anym.mean():.2f}")
        return M
    show("control, true AR1", out[len(real):])
    M = show("real correlated third", out[: len(real)])
    first = M.idxmax(axis=1).where(M.any(axis=1), "none")           # simplest formula that matches (MODELS order)
    print("   simplest matching formula:", (first.value_counts(normalize=True).round(2)).to_dict())
    print(f"total {time.time() - t0:.1f}s")
    wins = pd.Series([w for w, _ in out[: len(real)]])
    print("   BIC-best formula among the 'none' series:", wins[(first == "none").to_numpy()].value_counts(normalize=True).round(2).to_dict())
