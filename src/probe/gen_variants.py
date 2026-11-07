"""Held-out check of one generator variant (search best + overrides), for running many variants in parallel.
Same held-out real sample and fake seeds as probe/gen_validate.py.

    python probe/gen_variants.py NAME        # prints NAME, held-out mean KS and the per-stat KS
"""
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp

sys.path.insert(0, "probe")
import gen_search as gs
from ku_gen import DEFAULT, gen

V = {"V0 best": {}, "V1 catalog noise": dict(cat_noise=1), "V2 catalog AR1 phi": dict(cat_phi=1), "V3 catalog sine amp": dict(cat_sin=1),
     "V4 catalog all": dict(cat_noise=1, cat_phi=1, cat_sin=1), "V5 spikes .1": dict(spike_p=.1), "V6 spikes .2": dict(spike_p=.2),
     "V7 coupled .1": dict(couple_p=.1), "V8 GARCH b (.75,.95)": dict(garch_b=(.75, .95)), "V9 excursion .15": dict(excursion_p=.15),
     "V10 catalog+spikes .1": dict(cat_noise=1, cat_phi=1, cat_sin=1, spike_p=.1)}
name = sys.argv[1]
key = next(k for k in V if k.split()[0] == name)
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
first = set(L[L == 0].sample(2500, random_state=0).index) | set(L[L == 1].sample(2500, random_state=0).index)
rest = L[~L.index.isin(first)]
keep = set(rest[rest == 0].sample(2500, random_state=7).index) | set(rest[rest == 1].sample(2500, random_state=7).index)
try:
    R = pd.read_parquet("outputs/factory/real_stats_heldout.parquet")
except FileNotFoundError:
    S = pd.read_parquet("data/train_series.parquet"); S["id"] = S["id"].astype(str)
    S = S[S.id.isin(keep)].sort_values(["id", "time"])
    R = pd.DataFrame([dict(gs.st(g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())), y=L[k]) for k, g in S.groupby("id", sort=False)])
    R.to_parquet("outputs/factory/real_stats_heldout.parquet")
b = {k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v) for k, v in json.load(open("outputs/factory/ku_gen_best.json")).items()}
P = dict(DEFAULT, **b, **V[key])
ks = []
for y in (0, 1):
    xs = [gen(P, 500_000 + 2 * s + y, bool(y))[:2] for s in range(2500)]
    F = pd.DataFrame(Parallel(n_jobs=3, batch_size=64)(delayed(gs.st)(x, n0) for x, n0 in xs))
    r = R[R.y == y]
    ks += [ks_2samp(r[c].dropna(), F[c].dropna()).statistic for c in F.columns]
cols = list(F.columns)
worst = sorted(zip(ks, [f"{c}{y}" for y in (0, 1) for c in cols]), reverse=True)[:3]
print(f"{key:24s} held-out mean KS {np.mean(ks):.4f}  worst {[(n, round(k, 3)) for k, n in worst]}")
