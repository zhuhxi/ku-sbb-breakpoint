"""Out-of-sample check of the generator search (probe/gen_search.py): different fake seeds and a disjoint real sample
(2500 per class not used by the search cache). Compares DEFAULT, the search best and the old calib_break X1 menu.

    python probe/gen_validate.py
"""
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp

sys.path.insert(0, "probe")
import gen_search as gs
from ku_gen import DEFAULT, gen

L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
first = set(L[L == 0].sample(2500, random_state=0).index) | set(L[L == 1].sample(2500, random_state=0).index)
rest = L[~L.index.isin(first)]
keep = set(rest[rest == 0].sample(2500, random_state=7).index) | set(rest[rest == 1].sample(2500, random_state=7).index)
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
S = S[S.id.isin(keep)].sort_values(["id", "time"])
R = pd.DataFrame(Parallel(n_jobs=30)(delayed(lambda k, v, n0: dict(gs.st(v, n0), y=L[k]))(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum()))
                                     for k, g in S.groupby("id", sort=False)))
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
best = min((json.loads(l) for l in open(gs.OUT)), key=lambda d: d["obj"])["P"]
best = {k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v) for k, v in best.items()}
json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()}, open("outputs/factory/ku_gen_best.json", "w"), indent=1)
gens = {"DEFAULT": lambda s, y: gen(DEFAULT, s, y)[:2], "search best": lambda s, y: gen(best, s, y)[:2],
        "old X1": lambda s, y: cb.gen(cb.BREAKS["X1"], s, y, True)}
for name, g in gens.items():
    ks = []
    for y in (0, 1):
        xs = [g(500_000 + 2 * s + y, bool(y)) for s in range(2500)]
        F = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(gs.st)(x, n0) for x, n0 in xs))
        r = R[R.y == y]
        ks += [ks_2samp(r[c].dropna(), F[c].dropna()).statistic for c in F.columns]
    print(f"{name:12s} held-out mean KS {np.mean(ks):.4f}   per stat {np.round(ks, 3).tolist()}")
