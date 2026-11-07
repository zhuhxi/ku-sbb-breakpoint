"""Generator features for every train + test series, saved as factory families for confirm.py:
  outputs/factory/genA.parquet  structure-aware features (probe/gen_feats.py: structure, noise, same-structure LR test)
  outputs/factory/genB.parquet  break-size features (probe/model_feats.py: level in noise-sd, spread vs x2, dependence)

    python probe/genfeat_all.py [--jobs 64]
"""
import argparse
import importlib.util
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=int(os.environ.get("NTHREADS", 30)))
args = ap.parse_args()
sys.argv = [sys.argv[0]]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
gf = load("gf", "probe/gen_feats.py")
mf = load("mf", "probe/model_feats.py")


def both(v, n0):
    try:
        a = gf.gen_features(v, n0)
    except Exception:
        a = {}
    return a, mf.gfeats(v, n0)


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    keys, rows = zip(*[(k, (g["value"].to_numpy(dtype=float), int((g["period"] == 0).sum()))) for k, g in S.groupby("id")])
    out = Parallel(n_jobs=args.jobs, batch_size=32)(delayed(both)(v, n0) for v, n0 in rows)
    A = pd.DataFrame([a for a, _ in out], index=keys).astype(float)
    B = pd.DataFrame([b for _, b in out], index=keys).astype(float)
    A.to_parquet("outputs/factory/genA.parquet")
    B.to_parquet("outputs/factory/genB.parquet")
    print(f"genA {A.shape}, genB {B.shape}, missing genA rows {A.isna().all(axis=1).sum()}  ({time.time() - t0:.0f}s)")
