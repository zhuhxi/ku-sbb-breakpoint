"""probe/break_sizes_real.py on generated series (same filter and bins), real numbers alongside.

    python probe/break_sizes_fake.py MENU [N]
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()
menu = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24000
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
sys.argv = sys.argv[:1]
cb = load("cb", "probe/calib_break.py")
B = dict(cb.BREAKS[menu])
ac = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]


def one(s):
    x, n0 = cb.gen(B, 2 * 10 ** 6 + s, bool(s % 2), True)
    a, b = x[:n0], x[n0:]
    if abs(ac(a)) < .1 and abs(ac(b)) < .1:
        return dict(level=(b.mean() - a.mean()) / a.std(), spread=np.log(b.std() / a.std()), label=s % 2)


D = pd.DataFrame([r for r in Parallel(n_jobs=-1, batch_size=64)(delayed(one)(s) for s in range(N)) if r])
REAL = {"level": [-1142, -220, 134, 111, 198, 125, 74, -1, 36, 69, 166, 24, 29, 190, 173, 24, 9, 2],
        "spread": [-759, -357, -48, 65, 113, 142, 188, 161, 79, 59, 174, 157, 24, 1]}
print(f"{menu}: white-noise-like {len(D)} (label1 {int(D.label.sum())}; real 9115 / 4534)")
for c, edges in [("level", np.r_[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 5]),
                 ("spread", np.r_[0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .7, .8, 1, 1.5])]:
    k1 = (D.label == 1).sum()
    h1 = np.histogram(D.loc[D.label == 1, c].abs(), edges)[0]
    h0 = np.histogram(D.loc[D.label == 0, c].abs(), edges)[0] * k1 / (D.label == 0).sum()
    ex = (h1 - h0) * 4534 / k1                                  # scaled to the real label-1 count
    print(f"\n|{c}| bin     fake excess   real excess")
    for k in range(len(edges) - 1):
        print(f"  {edges[k]:4.2f}-{edges[k+1]:4.2f}   {ex[k]:+7.0f}     {REAL[c][k]:+6d}")
print(f"{time.time() - t0:.1f}s")
