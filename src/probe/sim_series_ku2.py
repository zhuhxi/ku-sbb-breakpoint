"""Write N fakes from probe/ku_gen.py with a parameter file, in the competition format, plus ALL hidden parameters.
    python probe/sim_series_ku2.py N PARAMS.json TAG SEED0  -> outputs/aug/series_TAG.parquet, outputs/aug/meta_TAG.parquet
"""
import json, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, "probe")
from ku_gen import DEFAULT, gen

N, PF, TAG, S0 = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
b = {k: (tuple(v) if isinstance(DEFAULT.get(k), tuple) else v) for k, v in json.load(open(PF)).items()}
P = dict(DEFAULT, **b)
out = Parallel(n_jobs=-1, batch_size=64)(delayed(gen)(P, S0 + s, bool(s % 2)) for s in range(N))
rows, meta = [], []
for s, (x, n0, info) in enumerate(out):
    n = len(x); i = f"{TAG}{s:06d}"
    rows.append(pd.DataFrame(dict(id=i, time=np.arange(n), value=x.astype(np.float64), period=(np.arange(n) >= n0).astype(np.int8))))
    meta.append(dict(id=i, label=s % 2, **info))
pd.concat(rows, ignore_index=True).to_parquet(f"outputs/aug/series_{TAG}.parquet")
pd.DataFrame(meta).set_index("id").to_parquet(f"outputs/aug/meta_{TAG}.parquet")
print(TAG, "series", N)
