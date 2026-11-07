"""Grid over the dependence-break sizes (weak range, strong size, strong share); spread fixed at the tuned values
(weak x1.1-1.35, strong x2 on 1/4), rest of the menu as 'weak-mostly unif2'. Scores the dep histogram L1 vs real
break series (no-break floor ~0.07).

    python probe/tune_dep.py
"""
import importlib.util
import itertools
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)
SPREAD = (list(np.round(np.linspace(.1, .3, 5), 3)) * 9 + [np.log(2)] * 15,)


def menu(weak, strong, ps):
    k = round(12 * ps)
    vals = list(np.round(np.linspace(*weak, 5), 3)) * (12 - k) + [strong] * (5 * k)
    return dict(cb.BREAKS["weak-mostly unif2"], spread=SPREAD, dep=(vals,))


if __name__ == "__main__":
    t0 = time.time()
    D = pd.read_parquet("outputs/probe/break_size.parquet")
    R1 = D.loc[D.y == 1, "dep"]
    out = []
    for weak, strong, ps in itertools.product([(.1, .25), (.15, .3), (.2, .35)], [.4, .5], [.1, .25]):
        S = pd.DataFrame(Parallel(n_jobs=30)(delayed(cb.gen)(menu(weak, strong, ps), 10 ** 6 + s) for s in range(1500)))
        S = S[S.cls.isin(["WN", "AR1", "AR2"])]
        out.append(dict(weak=weak, strong=strong, strong_share=ps, L1=round(cb.hist_l1(S.dep, R1, -.8, .8), 3)))
    print(pd.DataFrame(out).sort_values("L1").to_string(index=False))
    print(f"total {time.time() - t0:.1f}s")
