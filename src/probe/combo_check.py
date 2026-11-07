"""Fast check of a break menu by per-series measurements: share of change combinations (L = |level| > .75 noise-sd,
S = spread outside [0.8, 1.25], D = |dep| > .25) and big-level quantiles, real vs generated.

    python probe/combo_check.py MENU [MENU ...]
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

menus = sys.argv[1:]
sys.argv = [sys.argv[0]]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
bm = load("bm", "probe/break_measure.py")


def combo(T):
    f = pd.DataFrame({"L": T.level.abs() > 0.75, "S": (T.spread > 1.25) | (T.spread < 0.8), "D": T.dep.abs() > 0.25})
    return f.apply(lambda r: "+".join(c for c in "LSD" if r[c]) or "none", axis=1).value_counts(normalize=True).round(3)


def one(B, s):
    return bm.measure(*cb.gen(B, 4 * 10 ** 6 + s, True, True))


M = pd.read_parquet("outputs/probe/break_measure.parquet")
R = M[M.y == 1]
cols = {"real": combo(R)}
big = {"real": np.round(np.quantile(R.level.abs()[R.level.abs() > .75], [.5, .75, .9, .95, .99]), 2)}
for name in menus:
    G = pd.DataFrame(Parallel(n_jobs=30)(delayed(one)(cb.BREAKS[name], s) for s in range(4000)), columns=["level", "spread", "dep"])
    cols[name] = combo(G)
    big[name] = np.round(np.quantile(G.level.abs()[G.level.abs() > .75], [.5, .75, .9, .95, .99]), 2)
print(pd.DataFrame(cols).fillna(0).to_string())
for k, v in big.items():
    print(f"  big |level| q50/75/90/95/99 {k:10s} {v}")
