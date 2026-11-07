"""Grid over the spread-break sizes (weak range, strong size, strong share) with the rest of the break menu fixed at
'weak-mostly unif2'; scores the spread histogram L1 against real break series (no-break floor ~0.08).

    python probe/tune_spread.py
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
base = cb.BREAKS["weak-mostly unif2"]


def menu(weak, strong, ps):
    k = round(12 * ps)
    vals = list(np.round(np.linspace(*weak, 5), 3)) * (12 - k) + [strong] * (5 * k)
    return dict(base, spread=(vals,))


if __name__ == "__main__":
    t0 = time.time()
    D = pd.read_parquet("outputs/probe/break_size.parquet")
    R1 = D.loc[D.y == 1, "spread"]
    grid = list(itertools.product([(.05, .25), (.1, .3), (.15, .35), (0, .35)], [np.log(2)], [1 / 4, 1 / 3]))
    out = []
    for weak, strong, ps in grid:
        S = pd.DataFrame(Parallel(n_jobs=30)(delayed(cb.gen)(menu(weak, strong, ps), 10 ** 6 + s) for s in range(2500)))
        S = S[S.cls.isin(["WN", "AR1", "AR2"])]
        out.append(dict(weak_logk=weak, weak_x=f"{np.exp(weak[0]):.2f}-{np.exp(weak[1]):.2f}", strong_x=round(np.exp(strong), 2),
                        strong_share=round(ps, 2), L1=round(cb.hist_l1(S.spread, R1, -1.2, 1.2), 3)))
    print(pd.DataFrame(out).sort_values("L1").to_string(index=False))
    print(f"total {time.time() - t0:.1f}s")
