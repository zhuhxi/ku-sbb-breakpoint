"""Ceiling for shape breaks (noise redrawn after the boundary, probe/calib_corr.noise: normal / t(5-10) / two-level,
25% with GARCH-like clustering). Oracle: true structure, pre AR coefficients and outer shell are known, so the exact
innovations are recovered; two-sample statistics on them (KS, |log kurtosis ratio|, |log variance ratio|, change in
lag-1 autocorrelation of squares) and their rank sum. Practical: the same statistics on AR(2) residuals fitted on the
pre part. Stable structures only, no trend / drift.

    python probe/oracle_shape.py [MENU] [N]
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score

t0 = time.time()
menu = sys.argv[1] if len(sys.argv) > 1 else "ok4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
sys.argv = sys.argv[:1]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
B = cb.BREAKS[menu]
ST = ["ks", "kurt", "var", "arch"]


def stats(e, n0):
    a, b = e[:n0], e[n0:]
    k = lambda v: np.mean(((v - v.mean()) / v.std()) ** 4)
    q = lambda v: np.corrcoef((v ** 2)[1:], (v ** 2)[:-1])[0, 1]
    return [ks_2samp(a, b).statistic, abs(np.log(k(b) / k(a))), abs(np.log(b.var() / a.var())), abs(q(b) - q(a))]


def one(s):
    y = s % 2
    x, n0, d = cb.gen(B, 2 * 10 ** 6 + s, bool(y), "info")
    if d["structure"] not in ("WN", "AR1", "AR2") or d["trend"] or d["drift"] or (y and d["kind"] != "shape"):
        return None
    c = (x - d["base"]) / d["size"]
    p1, p2 = d["pre_phi"]
    e = c[2:] - p1 * c[1:-1] - p2 * c[:-2]
    X = np.column_stack([np.ones(n0 - 2), x[1:n0 - 1], x[:n0 - 2]])
    co = np.linalg.lstsq(X, x[2:n0], rcond=None)[0]
    r = x[2:] - co[0] - co[1] * x[1:-1] - co[2] * x[:-2]
    return [y] + stats(e, n0 - 2) + stats(r, n0 - 2)


D = pd.DataFrame([r for r in Parallel(n_jobs=-1, batch_size=64)(delayed(one)(s) for s in range(N)) if r],
                 columns=["y"] + [f"o_{c}" for c in ST] + [f"p_{c}" for c in ST])
for w in ("o", "p"):
    D[f"{w}_sum"] = sum(rankdata(D[f"{w}_{c}"]) for c in ST)
print(f"{menu}: shape-break positives {int(D.y.sum())}, negatives {int((D.y == 0).sum())}")
for c in ST + ["sum"]:
    print(f"  {c:5s} oracle {roc_auc_score(D.y, D['o_' + c]):.3f}   practical (pre-fit AR2 residuals) {roc_auc_score(D.y, D['p_' + c]):.3f}")
print(f"{time.time() - t0:.1f}s")
