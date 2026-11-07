"""Does a seasonal AR, x_t = a x_{t-1} + b x_{t-s} + e_t (one extra lag s), exactly match the ~9% of the correlated
third that no plain AR/MA formula matched? Control: true seasonal AR with the calibrated noise layer.

    python probe/exact_seas.py --n 2000
"""
import argparse
import importlib.util
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
args = ap.parse_args()
sys.argv = [sys.argv[0], "--n", "1"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
em = load("em", "probe/exact_match.py")
cc = load("cc", "probe/calib_corr.py")
mle = em.mle
M = 41


def seas(v):
    x = v - v.mean()
    n = len(x)
    best = None
    for s in range(2, M):
        X = np.column_stack([x[M - 1: n - 1], x[M - s: n - s]])
        b, *_ = np.linalg.lstsq(X, x[M:], rcond=None)
        e = x[M:] - X @ b
        if best is None or e.var() < best[0]:
            best = (e.var(), s, b, e)
    _, s, b, e = best
    return s, b, em.ljung_p(e) > 0.05


def one(v):
    _, win, res = mle.fit("x", v)
    none = not any(em.ljung_p(res[m]["e"]) > 0.05 for m in em.MODELS)
    s, b, ok = seas(v)
    return dict(win=win, none=none, s=s, a=b[0], b=b[1], ok=ok)


def ctrl(seed):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    s, a, b = int(r.integers(2, 41)), r.uniform(-0.4, 0.4), r.uniform(-0.5, 0.5)
    e, x = cc.noise(r, n), np.zeros(n)
    for t in range(s, n):
        x[t] = a * x[t - 1] + b * x[t - s] + e[t]
    return x


if __name__ == "__main__":
    t0 = time.time()
    R = np.load("outputs/probe/real_acf12_l0.npy")
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    ids0 = np.sort(lab.index[lab == 0].to_numpy())
    keep = ids0[(np.abs(R[:, 0]) >= 0.1) & (R[:, 0] <= 0.9)]
    ids = set(np.random.default_rng(0).choice(keep, args.n, replace=False))
    S = pd.read_parquet("data/train_series.parquet", columns=["id", "time", "value"])
    S = S[S.id.isin(ids)].sort_values(["id", "time"])
    D = pd.DataFrame(Parallel(n_jobs=30)(delayed(one)(g.to_numpy()) for _, g in S.groupby("id")["value"]))
    C = pd.DataFrame(Parallel(n_jobs=30)(delayed(seas)(ctrl(s)) for s in range(300)), columns=["s", "b", "ok"])
    print(f"control, true seasonal AR: matched {C.ok.mean():.2f}")
    rest = D[~D.win.str.startswith("SIN") & D.none]
    print(f"real unmatched 9% ({len(rest)} series): matched by seasonal AR {rest.ok.mean():.2f}")
    print("  best extra lag s (all unmatched):", rest.s.value_counts().head(8).to_dict())
    print("  coef on lag s, quantiles 10/50/90%:", np.round(np.quantile(rest.b, [.1, .5, .9]), 2))
    print(f"total {time.time() - t0:.1f}s")
