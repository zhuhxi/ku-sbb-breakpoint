"""Does 'sine + AR1' exactly match the ~9% of the correlated third that no plain AR/MA formula matched?
Control: true sine + AR1 series with the calibrated noise layer.

    python probe/exact_sinar.py --n 2000
"""
import argparse
import importlib.util
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
args = ap.parse_args()
sys.argv = [sys.argv[0], "--n", "1"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
em = load("em", "probe/exact_match.py")
cc = load("cc", "probe/calib_corr.py")
mle = em.mle


def one(v):
    _, win, res = mle.fit("x", v)
    ok = {m: em.ljung_p(res[m]["e"]) > 0.05 for m in em.MODELS + ["SIN+WN", "SIN+AR1"]}
    p = res["SIN+AR1"]
    return dict(win=win, none=not any(ok[m] for m in em.MODELS), sin_ar1=ok["SIN+AR1"], sin_wn=ok["SIN+WN"],
                period=1 / p["f"], amp=np.hypot(p["b"][1], p["b"][2]) / res["SIN+AR1"]["sigma"], phi=p["phi"][0])


def ctrl(seed):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    t = np.arange(n)
    x = lfilter([1], [1, -r.uniform(-0.85, 0.85)], cc.noise(r, n))
    return r.uniform(0.5, 3) * np.sin(2 * np.pi * t / r.integers(12, 33) + r.uniform(0, 6.3)) + x


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
    C = pd.DataFrame(Parallel(n_jobs=30)(delayed(one)(ctrl(s)) for s in range(400)))
    print(f"control, true sine + AR1: matched by sine+AR1 {C.sin_ar1.mean():.2f}")
    rest = D[~D.win.str.startswith("SIN") & D.none]
    print(f"real unmatched 9% ({len(rest)} series): matched by sine+AR1 {rest.sin_ar1.mean():.2f}, by sine+WN {rest.sin_wn.mean():.2f}")
    hit = rest[rest.sin_ar1]
    print("  matched ones: period", np.round(np.quantile(hit.period, [.1, .5, .9]), 1),
          " amp/noise", np.round(np.quantile(hit.amp, [.1, .5, .9]), 2), " phi", np.round(np.quantile(hit.phi, [.1, .5, .9]), 2))
    print(f"total {time.time() - t0:.1f}s")
