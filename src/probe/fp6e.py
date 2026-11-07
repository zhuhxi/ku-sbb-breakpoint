"""Which generated series have their best mean step far before the boundary (offset <= -51, probe/fp6c.py)?
Label-1 series of one menu, fp6 selection; broken down by the generator's structure / trend / drift layer.

    python probe/fp6e.py [MENU]
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()
menu = sys.argv[1] if len(sys.argv) > 1 else "L abc40 ns35 shell s6"
sys.argv = sys.argv[:1]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
src = open("probe/fp6c.py").read()
exec(src[src.index("BINS"):src.index("def fake")])
B = cb.BREAKS[menu]


def one(s):
    x, n0, info = cb.gen(B, 4 * 10 ** 6 + s, True, "info")
    o = off(x, n0)
    return None if o is None else dict(early=o <= -51, structure=info["structure"], trend=info["trend"], drift=info["drift"])


D = pd.DataFrame([r for r in Parallel(n_jobs=-1, batch_size=64)(delayed(one)(s) for s in range(16000)) if r])
print(menu, "selected label-1 series:", len(D), " early-step share", round(D.early.mean(), 3))
for c in ("structure", "trend", "drift"):
    T = D.groupby(c).early.agg(["size", "sum", "mean"]).rename(columns={"size": "series", "sum": "early", "mean": "early share"})
    print(T.round(2).to_string(), "\n")
print(f"{time.time() - t0:.1f}s")
