"""Binned before-vs-after change sizes: real no-break, real break, generated break (tuned menu), stationary pre only.

    python probe/break_bins.py
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.argv = [sys.argv[0], "--n", "1"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
c2 = load("c2", "probe/break_c2st.py")
D = pd.read_parquet("outputs/probe/break_size.parquet")
S1 = pd.DataFrame(Parallel(n_jobs=30)(delayed(c2.cb.gen)(c2.MENU, 10 ** 6 + s) for s in range(4000)))
S1 = S1[S1.cls.isin(["WN", "AR1", "AR2"])]
for c, bins in [("level", [-9, -2.5, -1.6, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 1.6, 2.5, 9]),
                ("spread", [-2, -0.6, -0.4, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.4, 0.6, 2]),
                ("dep", [-2, -0.4, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.4, 2])]:
    h = lambda z: np.histogram(z, bins)[0] / len(z)
    T = pd.DataFrame({"real0": h(D.loc[D.y == 0, c]), "real1": h(D.loc[D.y == 1, c]), "sim1": h(S1[c])}, index=[f"{a:+.2f}" for a in bins[:-1]])
    T["sim-real"] = T.sim1 - T.real1
    print(c, "(bin lower edge)"); print((100 * T).round(1).T.to_string())
