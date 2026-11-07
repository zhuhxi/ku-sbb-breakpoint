"""Structure variants of the clean generator (search-best numbers + one code-path switch), held-out mean KS averaged
over two fake-seed sets (objective noise ~.003 with one set). Appends to outputs/factory/struct_variants.jsonl.

    python probe/struct_variants.py NAME [NAME ...]
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

V = {"base": {}, "spread_out": dict(spread_target="output"), "shape_heavy": dict(shape_mode="heavy"), "shape_flip": dict(shape_mode="flip"),
     "decoy_any": dict(decoy_where="anywhere"), "decoy_brk": dict(decoy_in_breaks=1), "trend_all": dict(trend_fams="all"),
     "ll_arma": dict(ll_mode="arma"), "burn0": dict(burn=0), "sin_ar.3": dict(sin_ar_p=.3), "switch": dict(kind_mode="switch"),
     "lvl_series": dict(lvl_unit="series"), "icp0": dict(lvl_icp=0.0), "icp1": dict(lvl_icp=1.0), "renorm0": dict(dep_renorm=0.0),
     "nb_decoy": dict(nuis_brk=("decoy",)), "nb_vburst": dict(nuis_brk=("vburst",)), "nb_drift": dict(nuis_brk=("drift",)),
     "nb_dv": dict(nuis_brk=("decoy", "vburst")), "sw.15": dict(kind_mode="switch", switch_k=.15), "sw.6": dict(kind_mode="switch", switch_k=.6),
     "sw1": dict(kind_mode="switch", switch_k=1.0)}
R = pd.read_parquet("outputs/factory/real_stats_heldout.parquet")
b = {k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v) for k, v in json.load(open("outputs/factory/ku_gen_best.json")).items()}
import os
SEEDS = [int(v) for v in os.environ.get("SV_SEEDS", "500000,600000").split(",")]
OUTF = os.environ.get("SV_OUT", "outputs/factory/struct_variants.jsonl")
for name in sys.argv[1:]:
    P = dict(DEFAULT, **b)
    for part in name.split("+"):                       # combine switches with "+"
        P.update(V[part])
    allks = []
    for sd in SEEDS:
        ks = []
        for y in (0, 1):
            xs = [gen(P, sd + 2 * s + y, bool(y))[:2] for s in range(2500)]
            F = pd.DataFrame(Parallel(n_jobs=2, batch_size=64)(delayed(gs.st)(x, n0) for x, n0 in xs))
            r = R[R.y == y]
            ks += [ks_2samp(r[c].dropna(), F[c].dropna()).statistic for c in F.columns]
        allks.append(ks)
    ks = np.mean(allks, 0)
    cols = list(F.columns)
    rec = dict(name=name, ks=float(np.mean(ks)), per={f"{c}{y}": round(float(k), 3) for k, (y, c) in zip(ks, [(y, c) for y in (0, 1) for c in cols])})
    with open(OUTF, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{name:12s} held-out mean KS {rec['ks']:.4f}", flush=True)
