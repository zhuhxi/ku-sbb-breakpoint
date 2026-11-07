"""Verify 'simple formula + straight-line trend' for the ~9% unmatched part of the correlated third: remove a fitted
straight line, then re-run the exact match (residuals pure noise). Control: true AR1 + linear trend, same procedure.

    python probe/exact_trend.py
"""
import importlib.util
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter

warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0], "--n", "1"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
em = load("em", "probe/exact_match.py")
cc = load("cc", "probe/calib_corr.py")


def detrended_match(v):
    t = np.arange(len(v))
    b = np.polyfit(t, v, 1)
    win, ok = em.one(v - np.polyval(b, t))
    _, ok0 = em.one(v)
    return dict(before=any(ok0.values()), after=any(ok.values()), win=win,
                slope_total=b[0] * len(v) / np.std(v - np.polyval(b, t)))   # rise over the whole series, in noise sd


def ctrl(seed):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    return lfilter([1], [1, -r.uniform(-0.85, 0.85)], cc.noise(r, n)) + r.uniform(-3, 3) * np.arange(n) / n


if __name__ == "__main__":
    t0 = time.time()
    R = np.load("outputs/probe/real_acf12_l0.npy")
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    ids0 = np.sort(lab.index[lab == 0].to_numpy())
    keep = ids0[(np.abs(R[:, 0]) >= 0.1) & (R[:, 0] <= 0.9)]
    ids = set(np.random.default_rng(0).choice(keep, 2000, replace=False))
    S = pd.read_parquet("data/train_series.parquet", columns=["id", "time", "value"])
    S = S[S.id.isin(ids)].sort_values(["id", "time"])
    V = [g.to_numpy() for _, g in S.groupby("id")["value"]]
    fits = Parallel(n_jobs=30)(delayed(em.one)(v) for v in V)
    rest = [v for v, (w, ok) in zip(V, fits) if not w.startswith("SIN") and not any(ok.values())]
    C = pd.DataFrame(Parallel(n_jobs=30)(delayed(detrended_match)(ctrl(s)) for s in range(300)))
    D = pd.DataFrame(Parallel(n_jobs=30)(delayed(detrended_match)(v) for v in rest))
    print(f"control, true AR1 + line: matched before {C.before.mean():.2f} -> after removing the line {C.after.mean():.2f}")
    print(f"real unmatched ({len(D)}):  matched before {D.before.mean():.2f} -> after removing the line {D.after.mean():.2f}")
    h = D[D.after]
    print("  formula after removing the line:", h.win.value_counts().to_dict())
    print("  total rise of the line / noise sd, quantiles 10/50/90%:", np.round(np.quantile(np.abs(h.slope_total), [.1, .5, .9]), 2))
    print(f"total {time.time() - t0:.1f}s")
