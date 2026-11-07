"""What are the ~9% of the correlated third that no simple formula matches exactly? Compare against a control of true
AR1 series driven by the calibrated noise layer (t / bimodal / GARCH), and look for higher AR order, a hidden sine,
GARCH, and length.

    python probe/exact_rest.py --n 2000
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


def info(v):
    win, ok = em.one(v)
    n = len(v)
    x = v - v.mean()
    bic = []
    for p in range(1, 11):                                  # does a higher AR order fix it?
        X = np.column_stack([x[p - k - 1: n - k - 1] for k in range(p)])
        b, *_ = np.linalg.lstsq(X, x[p:], rcond=None)
        e = x[p:] - X @ b
        bic.append((n * np.log(e.var()) + p * np.log(n), p, em.ljung_p(e) > 0.05))
    _, pbest, okp = min(bic)
    pw = np.abs(np.fft.rfft(x)) ** 2
    e1 = x[1:] - (x[1:] @ x[:-1]) / (x[:-1] @ x[:-1]) * x[:-1]
    a = e1 ** 2 - (e1 ** 2).mean()
    return dict(win=win, none=not any(ok.values()), n=n, p_best=pbest, ar_p_ok=okp,
                peak=pw[1:].max() / np.median(pw[1:]), sq_acf1=(a[1:] @ a[:-1]) / (a @ a),
                kurt=((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3)


def ctrl(seed):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    return lfilter([1], [1, -r.uniform(-0.85, 0.85)], cc.noise(r, n))


if __name__ == "__main__":
    t0 = time.time()
    R = np.load("outputs/probe/real_acf12_l0.npy")
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    ids0 = np.sort(lab.index[lab == 0].to_numpy())
    keep = ids0[(np.abs(R[:, 0]) >= 0.1) & (R[:, 0] <= 0.9)]
    ids = set(np.random.default_rng(0).choice(keep, args.n, replace=False))
    S = pd.read_parquet("data/train_series.parquet", columns=["id", "time", "value"])
    S = S[S.id.isin(ids)].sort_values(["id", "time"])
    D = pd.DataFrame(Parallel(n_jobs=30)(delayed(info)(g.to_numpy()) for _, g in S.groupby("id")["value"]))
    C = pd.DataFrame(Parallel(n_jobs=30)(delayed(info)(ctrl(s)) for s in range(600)))
    print(f"control: true AR1 + calibrated noise -> unmatched {C.none.mean():.3f}")
    nonsin = ~D.win.str.startswith("SIN")
    rest, good = D[nonsin & D.none], D[nonsin & ~D.none]
    print(f"real: non-sine {nonsin.sum()}, unmatched {len(rest)} ({len(rest) / nonsin.sum():.3f})")
    cols = ["n", "p_best", "ar_p_ok", "peak", "sq_acf1", "kurt"]
    print(pd.DataFrame({"unmatched": rest[cols].median(), "matched": good[cols].median(),
                        "unmatched mean": rest[cols].mean(), "matched mean": good[cols].mean()}).round(3).to_string())
    print("unmatched, BIC-best AR order:", rest.p_best.value_counts().sort_index().to_dict())
    print(f"total {time.time() - t0:.1f}s")
