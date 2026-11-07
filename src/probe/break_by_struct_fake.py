"""probe/break_by_struct.py table on generated series, grouped by the TRUE generator structure
(WN/AR1/AR2 -> stable, SIN, RW, LL -> 'ARMA11/local level'; the real table groups by the fitted structure).

    python probe/break_by_struct_fake.py MENU [N] [--fit]   # --fit: group by the fitted structure (probe/gen_feats.py), as the real table
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

t0 = time.time()
fit = "--fit" in sys.argv
sys.argv = [a for a in sys.argv if a != "--fit"]
menu = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
sys.argv = sys.argv[:1]
src = open("probe/break_by_struct.py").read()
exec(src[src.index("def ac1"):src.index("D = pd.DataFrame")])     # ac1, peak, meas
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
B = dict(cb.BREAKS[menu])
if fit:
    spec = importlib.util.spec_from_file_location("gf", "probe/gen_feats.py"); gf = importlib.util.module_from_spec(spec); spec.loader.exec_module(gf)
GF = {0: "stable", 1: "stable", 2: "stable", 3: "stable", 4: "ARMA11/local level", 5: "RW", 6: "SIN", 7: "SIN"}
G = {"WN": "stable", "AR1": "stable", "AR2": "stable", "SIN": "SIN", "RW": "RW", "LL": "ARMA11/local level"}


def one(s):
    x, n0, info = cb.gen(B, 6 * 10 ** 6 + s, bool(s % 2), "info")
    grp = GF[gf.gen_features(x, n0)["g_struct"]] if fit else G[info["structure"]]
    return dict(meas(x, n0), y=s % 2, grp=grp, true=info["structure"])


D = pd.DataFrame(Parallel(n_jobs=-1, batch_size=16)(delayed(one)(s) for s in range(N)))
rows = {}
for grp, E in D.groupby("grp"):
    rows[f"{grp} (n={len(E)})"] = {c: roc_auc_score(E.y, E[c].astype(float).fillna(0)) for c in D.columns if c not in ("y", "grp", "true")}
print(menu, "grouped by", "fitted" if fit else "true", "structure")
if fit:
    print(pd.crosstab(D.true, D.grp).to_string())
print(pd.DataFrame(rows).T.round(3).to_string())
print(f"{time.time() - t0:.1f}s")
