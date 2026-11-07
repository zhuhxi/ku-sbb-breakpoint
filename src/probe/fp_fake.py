"""Fingerprints 1-3 (probe/fingerprints.py) on generated series, as the baseline for the real numbers.
 ours    : our generator (recursion from 0, one continuous recursion across the boundary)
 restart : label-1 series whose after-part is a fresh series started from 0 (pasted on)

    python probe/fp_fake.py [N]
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
sys.argv = sys.argv[:1]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
B = cb.BREAKS["L abc40 ns35 shell s6"]
z = lambda v: (v - v.mean()) / v.std()


def meas(x, n0, y, kind):
    a, b = z(x[:n0]), z(x[n0:])
    return dict(kind=kind, y=y, ac=np.mean(a[1:] * a[:-1]), inside=np.mean(a[-11:-1] * a[-10:]), cross=a[-1] * b[0],
                start=np.mean(a[:5] ** 2), later=np.mean(a[30:35] ** 2), bstart=np.mean(b[:5] ** 2), blater=np.mean(b[30:35] ** 2))


def one(s):
    y = s % 2
    x, n0 = cb.gen(B, 8 * 10 ** 6 + s, bool(y), True)
    out = [meas(x, n0, y, "ours")]
    if y:
        x2, _ = cb.gen(B, 9 * 10 ** 6 + s, False, True)
        if len(x2) >= len(x) - n0:
            out.append(meas(np.r_[x[:n0], x2[:len(x) - n0]], n0, 1, "restart"))
    return out


D = pd.DataFrame([d for o in Parallel(n_jobs=-1, batch_size=32)(delayed(one)(s) for s in range(N)) for d in o])
D = D[D.ac > .5]
print(D.groupby(["kind", "y"])[["ac", "inside", "cross", "start", "later", "bstart", "blater"]].mean().round(3).to_string())
print("counts", D.groupby(["kind", "y"]).size().to_dict(), f"  {time.time() - t0:.1f}s")
