"""ClaSP-style scores (probe/clasp_feats.py) for every train + test series, saved as a factory family for confirm.py:
outputs/factory/clasp.parquet.

    python probe/clasp_all.py [--jobs 64]
"""
import argparse
import importlib.util
import os
import time
import warnings

import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=int(os.environ.get("NTHREADS", 30)))
args = ap.parse_args()
s = importlib.util.spec_from_file_location("cf", "probe/clasp_feats.py")
cf = importlib.util.module_from_spec(s); s.loader.exec_module(cf)

if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    keys, rows = zip(*[(k, (g["value"].to_numpy(dtype=float), int((g["period"] == 0).sum()))) for k, g in S.groupby("id")])
    C = pd.DataFrame(Parallel(n_jobs=args.jobs, batch_size=32)(delayed(cf.clasp)(v, n0) for v, n0 in rows), index=keys).astype(float)
    C.to_parquet("outputs/factory/clasp.parquet")
    print(f"clasp {C.shape}, missing {C.isna().any(axis=1).sum()}  ({time.time() - t0:.0f}s)")
