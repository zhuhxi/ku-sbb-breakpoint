"""Ceiling for dependence breaks: an oracle that knows everything except whether there is a break (structure, pre-break
AR coefficients, outer shell base / size, unit innovation variance) computes the likelihood ratio of
"pre coefficients throughout" vs "lag-1 coefficient free after the boundary" (lag-2 kept). Stable structures only
(WN / AR1 / AR2), no trend / drift layer. Compared with a plain |lag-1 autocorrelation change|.

    python probe/oracle_dep.py [MENU] [N]
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

t0 = time.time()
menu = sys.argv[1] if len(sys.argv) > 1 else "ok4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
sys.argv = sys.argv[:1]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
B = cb.BREAKS[menu]


def one(s):
    y = s % 2
    x, n0, d = cb.gen(B, 2 * 10 ** 6 + s, bool(y), "info")
    if d["structure"] not in ("WN", "AR1", "AR2") or d["trend"] or d["drift"] or (y and d["kind"] != "dep"):
        return None
    c = (x - d["base"]) / d["size"]
    p1, p2 = d["pre_phi"]
    t = np.arange(max(n0, 2), len(c))
    yy, z = c[t] - p2 * c[t - 2], c[t - 1]
    sse0 = np.sum((yy - p1 * z) ** 2)
    sse1 = sse0 - (z @ (yy - p1 * z)) ** 2 / (z @ z)          # best lag-1 coefficient after the boundary
    a, b = c[:n0], c[n0:]
    ac = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]
    return dict(y=y, oracle=sse0 - sse1, plain=abs(ac(b) - ac(a)), size=d["dep"])


D = pd.DataFrame([r for r in Parallel(n_jobs=-1, batch_size=64)(delayed(one)(s) for s in range(N)) if r])
pos, neg = D[D.y == 1], D[D.y == 0]
au = lambda P, c: roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(P))], np.r_[neg[c], P[c]])
print(f"{menu}: dependence-break positives {len(pos)}, negatives {len(neg)}")
print(f"  all   oracle {au(pos, 'oracle'):.3f}   plain |d acf1| {au(pos, 'plain'):.3f}")
for g, P in pos.groupby(pd.qcut(pos["size"], 3, labels=["small", "mid", "large"]), observed=True):
    print(f"  {g:5s} (median |delta phi| {P['size'].median():.3f})  oracle {au(P, 'oracle'):.3f}   plain {au(P, 'plain'):.3f}")
print(f"{time.time() - t0:.1f}s")
