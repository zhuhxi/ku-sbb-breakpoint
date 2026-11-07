"""Same per-series measurements (probe/break_measure.py) on generated break series from a menu, side by side with
real break series. Stationary pre only.

    python probe/break_measure_cmp.py MENU
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

name = sys.argv[1]
sys.argv = [sys.argv[0]]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
bm = load("bm", "probe/break_measure.py")


def one(s):
    x, n0 = cb.gen(cb.BREAKS[name], 3 * 10 ** 6 + s, True, True)
    return bm.measure(x, n0)


M = pd.read_parquet("outputs/probe/break_measure.parquet")
G = pd.DataFrame(Parallel(n_jobs=30)(delayed(one)(s) for s in range(4000)), columns=["level", "spread", "dep"])
pd.set_option("display.width", 250)
for c, bins in [("level", np.r_[-9, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 9]),
                ("spread", np.r_[0, 0.45, 0.55, 0.65, 0.8, 0.9, 1.1, 1.25, 1.6, 1.8, 2.2, 9]),
                ("dep", np.r_[-2, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 2])]:
    h = lambda z: np.histogram(z, bins)[0] / len(z) * 100
    T = pd.DataFrame({"real": h(M.loc[M.y == 1, c]), "gen": h(G[c])}, index=[f"{a:g}" for a in bins[:-1]])
    T["gen-real"] = T.gen - T.real
    print(f"{c}"); print(T.round(1).T.to_string())
