"""Write generated series (calibrated no-break + break menu) in the competition format for augmentation.

    python probe/sim_series.py N [MENU]   # -> outputs/aug/series_sim.parquet, outputs/aug/meta_sim.parquet

Half label 0, half label 1, same generator (probe/calib_break.gen) so only the break differs.
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

N = int(sys.argv[1])
menu = sys.argv[2] if len(sys.argv) > 2 else "L abc40 ns35"
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
B = dict(cb.BREAKS[menu])
lab = np.arange(N) % 2
out = Parallel(n_jobs=-1)(delayed(cb.gen)(B, 9 * 10 ** 6 + s, bool(lab[s]), True) for s in range(N))
rows = []
for s, (x, n0) in enumerate(out):
    n = len(x)
    rows.append(pd.DataFrame(dict(id=f"sim{s:06d}", time=np.arange(n), value=x.astype(np.float64),
                                  period=(np.arange(n) >= n0).astype(np.int8))))
pd.concat(rows, ignore_index=True).to_parquet("outputs/aug/series_sim.parquet")
pd.DataFrame(dict(parent=None, label=lab), index=[f"sim{s:06d}" for s in range(N)]).rename_axis("id").to_parquet("outputs/aug/meta_sim.parquet")
print("sim series", N, "menu", menu)
