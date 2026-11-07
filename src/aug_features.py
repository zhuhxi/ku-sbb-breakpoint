"""combo10 feature columns for the augmented series (same code as the originals).

    python aug_features.py crop      # -> outputs/aug/X_crop.parquet
    python aug_features.py spike     # -> outputs/aug/X_spike.parquet

Builds features.py, harvest, magic B and the factory families of combo10 (st_RES10, st_ARES10, fe, fe2, ar2,
cost). For spike rows the magic B columns are copied from the parent series.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import factory  # noqa: E402
import harvest  # noqa: E402
from brute.magic_b import magic_b  # noqa: E402
from features import build_features  # noqa: E402

kind = sys.argv[1]
OUT = Path("outputs/aug")
path = OUT / f"series_{kind}.parquet"
nj = int(os.environ.get("NTHREADS", 30))
t = time.time()
F = build_features(pd.read_parquet(path), n_jobs=nj)
print("features", F.shape, f"{time.time() - t:.0f}s", flush=True)
H = harvest.build(str(path))
print("harvest", H.shape, f"{time.time() - t:.0f}s", flush=True)
if kind == "spike":
    meta = pd.read_parquet(OUT / "meta.parquet")
    Bp = pd.read_parquet("outputs/magic_b_train.parquet")
    B = Bp.reindex(meta.loc[F.index, "parent"].to_numpy())
    B.index = F.index
else:
    B = magic_b(str(path))
print("magic_b", B.shape, flush=True)

d = pd.read_parquet(path).sort_values(["id", "time"], kind="stable")
items = [(sid, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for sid, g in d.groupby("id", sort=False)]
ALL = {**factory.TRANSFORMS, **factory.EXTRA_TRANSFORMS}
parts = []
for fam, trs in (("st", ["RES10"]), ("st", ["ARES10"]), ("fe", None), ("fe2", None), ("ar2", None), ("cost", None)):
    tr = {k: ALL[k] for k in trs} if trs else factory.TRANSFORMS
    res = Parallel(n_jobs=nj)(delayed(factory.run_chunk)([fam], items[i::nj * 4], tr) for i in range(nj * 4))
    parts.append(pd.DataFrame({k: v for r in res for k, v in r.items()}).T)
    print(fam, trs, parts[-1].shape, f"{time.time() - t:.0f}s", flush=True)
N = pd.concat(parts, axis=1)
N = N.loc[:, ~N.columns.duplicated()]
X = pd.concat([F, B.reindex(F.index), H.reindex(F.index), N.reindex(F.index)], axis=1)
X = X.loc[:, ~X.columns.duplicated()].replace([np.inf, -np.inf], np.nan).astype(np.float32)
X.to_parquet(OUT / f"X_{kind}.parquet")
print("done", X.shape, f"{time.time() - t:.0f}s")
