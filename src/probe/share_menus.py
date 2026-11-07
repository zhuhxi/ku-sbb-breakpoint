"""Which structure-share menu looks most like code someone wrote? Simulate each true structure on pre-boundary
lengths, fit with the same MLE picker (confusion matrix), then predicted fitted shares = p @ C vs the real
pre-part fitted shares (probe/pre_struct_label.py, 4000 series).

    python probe/share_menus.py [PER]
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
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 500
sys.argv = sys.argv[:1]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
gf = load("gf", "probe/gen_feats.py")
cc = load("cc", "probe/calib_corr.py")
TRUE = ["WN", "AR1", "AR2", "SIN", "RW", "LL"]
FIT = ["WN", "AR1", "AR2", "MA1", "ARMA11", "RW", "SIN+WN", "SIN+AR1"]
REAL = pd.Series(dict(WN=1926, AR1=502, AR2=205, MA1=169, ARMA11=146, RW=300, **{"SIN+WN": 724, "SIN+AR1": 28}))[FIT]
n0s = pd.read_parquet("/tmp/claude-1006/-mnt-disk2-zhuhx-yolov10/52e90985-5ae5-4402-bf67-8e7f98c2f063/scratchpad/nn0_train.parquet").n0.to_numpy()


def sim(k, s):                                       # calib_params.gen menu F structures, on a real pre-part length
    r = np.random.default_rng(4 * 10 ** 6 + s)
    n = int(r.choice(n0s)); e, t = cc.noise(r, n), np.arange(n)
    if k == "WN": x = e
    elif k == "SIN": x = r.uniform(0.5, 3) * np.sin(2 * np.pi * t / r.integers(12, 33) + r.uniform(0, 6.3)) + e
    elif k == "RW": x = np.cumsum(e)
    elif k == "AR1": x = lfilter([1], [1, -r.uniform(-.7, .7)], e)
    elif k == "AR2":
        while True:
            p1, p2 = r.uniform(-.8, .8), r.uniform(-.6, .6)
            if p2 + p1 < 1 and p2 - p1 < 1: break
        x = lfilter([1], [1, -p1, -p2], e)
    else: x = np.cumsum(cc.noise(r, n) * r.uniform(.02, .1)) + e
    if k in ("WN", "AR1", "AR2") and r.random() < .08:
        x = x + r.choice([-1, 1]) * r.uniform(.5, 2.5) * x.std() * t / n
    return k, gf.mle.fit("b", x)[1]


out = Parallel(n_jobs=-1, batch_size=8)(delayed(sim)(k, i * 6 + j) for i in range(PER) for j, k in enumerate(TRUE))
C = pd.crosstab(pd.Series([a for a, _ in out]), pd.Series([b for _, b in out]), normalize="index").reindex(index=TRUE, columns=FIT).fillna(0)
print("confusion P(fitted | true):\n", C.round(2).to_string())
real = REAL / REAL.sum()
MENUS = {
    "our estimate .42/.17/.10/.17/.07/.07": [.42, .17, .10, .17, .07, .07],
    "equal 1/6": [1 / 6] * 6,
    "round .4/.2/.1/.2/.05/.05": [.4, .2, .1, .2, .05, .05],
    "round .4/.15/.15/.15/.075/.075": [.4, .15, .15, .15, .075, .075],
    "round .4/.2/.1/.15/.1/.05": [.4, .2, .1, .15, .1, .05],
    "round .5/.1/.1/.2/.05/.05": [.5, .1, .1, .2, .05, .05],
    "two-level stable.6(=)/sin.2/drift.2(=)": [.2, .2, .2, .2, .1, .1],
    "two-level stable.7(=)/sin.15/drift.15(=)": [.7 / 3] * 3 + [.15, .075, .075],
    "WN half: .5/.125/.125/.125/.0625/.0625": [.5, .125, .125, .125, .0625, .0625],
}
ll = lambda p: -(REAL.to_numpy() * np.log(np.clip(np.asarray(p) @ C.to_numpy(), 1e-9, None))).sum()
best = minimize(lambda z: ll(np.exp(z) / np.exp(z).sum()), np.zeros(6), method="Nelder-Mead", options=dict(maxiter=4000)).x
pb = np.exp(best) / np.exp(best).sum()
MENUS["best fit (free)"] = list(pb)
print("\nbest-fit shares", dict(zip(TRUE, np.round(pb, 3))))
base = ll(pb)
print("\nmenu                                         loglik gap vs best (<2 = as good as best)   L1 of fitted shares")
for name, p in sorted(MENUS.items(), key=lambda kv: ll(kv[1])):
    print(f"  {name:44s} {ll(p) - base:8.1f}   {np.abs(np.asarray(p) @ C.to_numpy() - real.to_numpy()).sum():.3f}")
print(f"{time.time() - t0:.1f}s")
