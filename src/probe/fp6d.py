"""Fingerprint 6 tables (probe/fp6b.py kinds, probe/fp6c.py step offsets) on label-1 series from generator menus,
to compare with the real label-1 rows printed by those two scripts.

    python probe/fp6d.py MENU [MENU ...]
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()
menus = sys.argv[1:]
sys.argv = sys.argv[:1]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
for f in ("probe/fp6b.py", "probe/fp6c.py"):                  # fit() and off() without running the scripts
    src = open(f).read()
    exec(src[src.index("G = 150") if "G = 150" in src else src.index("BINS"):src.index("def fake")])
N = 10000


def one(B, s):
    x, n0 = cb.gen(B, 4 * 10 ** 6 + s, True, True)
    k = fit(x, n0)
    return None if k is None else (k, off(x, n0))


for m in menus:
    R = pd.DataFrame([r for r in Parallel(n_jobs=-1, batch_size=64)(delayed(one)(cb.BREAKS[m], s) for s in range(N)) if r], columns=["kind", "offset"])
    a = R.kind.value_counts(normalize=True).round(2).to_dict()
    b = pd.cut(R.offset, BINS).value_counts(normalize=True, sort=False).round(2).to_dict()
    print(f"{m:24s} n={len(R)}  kinds {a}\n{'':24s} offsets {dict((str(k), v) for k, v in b.items())}   {time.time() - t0:.1f}s")
