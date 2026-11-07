"""Robustness of the structure-share estimate (probe/share_menus.py): several settings, and MA1 / ARMA11 added as
possible true structures. Prints the free best-fit shares per setting.

    python probe/share_robust.py [PER]
"""
import importlib.util
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.signal import lfilter

warnings.filterwarnings("ignore")
t0 = time.time()
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 300
sys.argv = sys.argv[:1]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
gf = load("gf", "probe/gen_feats.py")
cc = load("cc", "probe/calib_corr.py")
FIT = ["WN", "AR1", "AR2", "MA1", "ARMA11", "RW", "SIN+WN", "SIN+AR1"]
REAL = np.array([1926, 502, 205, 169, 146, 300, 724, 28])
n0s = pd.read_parquet("/tmp/claude-1006/-mnt-disk2-zhuhx-yolov10/52e90985-5ae5-4402-bf67-8e7f98c2f063/scratchpad/nn0_train.parquet").n0.to_numpy()
SET = {"base (AR1 .7, normal/t/bimodal noise)": dict(a1=.7, a2=((-.8, .8), (-.6, .6)), noise="mix"),
       "wide AR (AR1 .85, AR2 1.5/.8)": dict(a1=.85, a2=((-1.5, 1.5), (-.8, .8)), noise="mix"),
       "narrow AR (AR1 .5)": dict(a1=.5, a2=((-.6, .6), (-.4, .4)), noise="mix"),
       "normal noise only": dict(a1=.7, a2=((-.8, .8), (-.6, .6)), noise="normal")}


def sim(k, s, P):
    r = np.random.default_rng(3 * 10 ** 6 + s)
    n = int(r.choice(n0s)); t = np.arange(n)
    e = cc.noise(r, n) if P["noise"] == "mix" else r.standard_normal(n)
    if k == "WN": x = e
    elif k == "SIN": x = r.uniform(0.5, 3) * np.sin(2 * np.pi * t / r.integers(12, 33) + r.uniform(0, 6.3)) + e
    elif k == "RW": x = np.cumsum(e)
    elif k == "AR1": x = lfilter([1], [1, -r.uniform(-P["a1"], P["a1"])], e)
    elif k == "AR2":
        while True:
            p1, p2 = r.uniform(*P["a2"][0]), r.uniform(*P["a2"][1])
            if p2 + p1 < 1 and p2 - p1 < 1: break
        x = lfilter([1], [1, -p1, -p2], e)
    elif k == "MA1": x = lfilter([1, r.uniform(-.7, .7)], [1], e)
    elif k == "ARMA11": x = lfilter([1, r.uniform(-.7, .7)], [1, -r.uniform(-P["a1"], P["a1"])], e)
    else: x = np.cumsum(cc.noise(r, n) * r.uniform(.02, .1)) + e
    return k, gf.mle.fit("b", x)[1]


for TRUE in (["WN", "AR1", "AR2", "SIN", "RW", "LL"], ["WN", "AR1", "AR2", "SIN", "RW", "LL", "MA1", "ARMA11"]):
    print(f"\ntrue structures allowed: {TRUE}")
    for name, P in SET.items():
        out = Parallel(n_jobs=-1, batch_size=8)(delayed(sim)(k, i * 8 + j, P) for i in range(PER) for j, k in enumerate(TRUE))
        C = pd.crosstab(pd.Series([a for a, _ in out]), pd.Series([b for _, b in out]), normalize="index").reindex(index=TRUE, columns=FIT).fillna(0).to_numpy()
        ll = lambda p: -(REAL * np.log(np.clip(p @ C, 1e-9, None))).sum()
        z = minimize(lambda z: ll(np.exp(z) / np.exp(z).sum()), np.zeros(len(TRUE)), method="Nelder-Mead", options=dict(maxiter=6000)).x
        p = np.exp(z) / np.exp(z).sum()
        print(f"  {name:40s}", " ".join(f"{k} {v:.2f}" for k, v in zip(TRUE, p)), f"| L1 {np.abs(p @ C - REAL / REAL.sum()).sum():.3f}")
print(f"{time.time() - t0:.1f}s")
