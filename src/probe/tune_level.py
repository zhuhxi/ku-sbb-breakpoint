"""Grid over the level-break parameters (strong share, strong fixed 2.0 vs U[1.8, 2.2], weak upper bound), rest of the
menu as 'A drift8'. Score: L1 between the histograms of the measured level change (noise-sd units,
probe/break_measure.py) of generated vs real break series (stationary pre), same buckets as probe/c2st_by_level.py.

    python probe/tune_level.py
"""
import importlib.util
import itertools
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.argv = [sys.argv[0]]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
bm = load("bm", "probe/break_measure.py")
B = [0, .1, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2.25, 2.75, 3.5, 99]


def level_list(share, strong, weak_hi):
    weak = list(np.round(np.linspace(0.05, weak_hi, 10), 3))
    st = [2.0] * 5 if strong == "fixed" else list(np.round(np.linspace(1.8, 2.2, 5), 2))
    k = round(20 * share)
    return (weak * (20 - k) + st * 2 * k,)


def one(menu, s):
    x, n0 = cb.gen(menu, 12 * 10 ** 6 + s, True, True)
    return bm.measure(x, n0)[0]


if __name__ == "__main__":
    t0 = time.time()
    M = pd.read_parquet("outputs/probe/break_measure.parquet")
    hr = np.histogram(M.loc[M.y == 1, "level"].abs(), B)[0] / (M.y == 1).sum()
    out = []
    for share, strong, hi, pl in itertools.product([.1, .15, .2], ["fixed", "1.8-2.2"], [.3], [.3, .45]):
        base = cb.BREAKS["A drift8"]
        menu = dict(base, level=level_list(share, strong, hi), switches=dict(base["switches"], level=pl))
        L = np.abs(Parallel(n_jobs=30)(delayed(one)(menu, s) for s in range(2500)))
        hg = np.histogram(L, B)[0] / len(L)
        out.append(dict(strong_share=share, strong=strong, weak_hi=hi, level_p=pl, L1=round(np.abs(hg - hr).sum(), 3)))
    print(pd.DataFrame(out).sort_values("L1").head(8).to_string(index=False))
    print(f"total {time.time() - t0:.1f}s")
