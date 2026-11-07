"""Does the noise shape change at the boundary? Residuals of the pre-boundary AR(2) formula (with a step dummy), split
before / after; per side: excess kurtosis and best noise type (normal / t / bimodal by BIC, probe/close_loop.py).
Real no-break vs real break vs generated break (menu), with the before x after type table (% of series).

    python probe/noise_shape.py MENU --n 1500
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
ap.add_argument("menu")
ap.add_argument("--n", type=int, default=1500)
args = ap.parse_args()
sys.argv = [sys.argv[0]]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
cl = load("cl", "probe/close_loop.py")


def shape(v, n0):
    n = len(v)
    X = np.column_stack([np.ones(n - 2), (np.arange(2, n) >= n0).astype(float), v[1:-1], v[:-2]])
    b, *_ = np.linalg.lstsq(X, v[2:], rcond=None)
    e = v[2:] - X @ b
    ea, eb = e[: n0 - 2], e[n0 - 2:]
    ta, tb = cl.fit_noise(ea)[0], cl.fit_noise(eb)[0]
    return dict(k_before=stats.kurtosis(ea), k_after=stats.kurtosis(eb), type_before=ta, type_after=tb)


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    M = pd.read_parquet("outputs/probe/break_measure.parquet")          # stationary-pre ids
    S = pd.read_parquet("data/train_series.parquet")
    groups = {}
    for y in (0, 1):
        ids = set(np.random.default_rng(y).choice(M.index[M.y == y], args.n, False))
        groups[f"real {'break' if y else 'no-break'}"] = [(g["value"].to_numpy(), int((g["period"] == 0).sum()))
                                                          for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
    groups[f"gen break ({args.menu})"] = Parallel(n_jobs=30)(delayed(cb.gen)(cb.BREAKS[args.menu], 6 * 10 ** 6 + s, True, True) for s in range(args.n))
    groups["gen no-break"] = Parallel(n_jobs=30)(delayed(cb.gen)(None, 6 * 10 ** 6 + s, False, True) for s in range(args.n))
    for name, L in groups.items():
        D = pd.DataFrame(Parallel(n_jobs=30)(delayed(shape)(v, n0) for v, n0 in L))
        dk = D.k_after - D.k_before
        sw = (D.type_before != D.type_after).mean()
        trans = (D.type_before + ">" + D.type_after)[D.type_before != D.type_after].value_counts(normalize=True).head(4).round(2).to_dict()
        print(f"{name:22s} kurt diff q10/50/90 {np.round(np.quantile(dk, [.1, .5, .9]), 2)}  |diff| q90 {np.quantile(dk.abs(), .9):.2f}"
              f"  noise type switches {sw:.3f}")
        print(pd.crosstab(D.type_before, D.type_after, normalize="all").mul(100).round(1).to_string(), "\n")
    print(f"total {time.time() - t0:.1f}s")
