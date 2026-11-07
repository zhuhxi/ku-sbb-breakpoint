"""Check "two-level (bimodal) noise is never switched at a break": share of generated break series that are bimodal
before the boundary and stay bimodal after (noise types as in probe/noise_shape.py), with and without the rule,
next to real break series (77%).

    python probe/keep_bimodal_check.py
"""
import sys

sys.argv = [sys.argv[0], "x"]
import importlib.util
import pandas as pd
from joblib import Parallel, delayed

s = importlib.util.spec_from_file_location("ns", "probe/noise_shape.py")
ns = importlib.util.module_from_spec(s); s.loader.exec_module(ns)
for menu in ("L abc40 ns35", "L abc40 ns35 kb"):
    L = Parallel(n_jobs=30)(delayed(ns.cb.gen)(ns.cb.BREAKS[menu], 6 * 10 ** 6 + i, True, True) for i in range(3000))
    D = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(ns.shape)(v, n0) for v, n0 in L))
    b = D[D.type_before == "bimodal"]
    print(f"{menu:18s} bimodal before {len(b) / len(D) * 100:.1f}%  stay bimodal after {(b.type_after == 'bimodal').mean() * 100:.0f}%   (real break: 77%)")
