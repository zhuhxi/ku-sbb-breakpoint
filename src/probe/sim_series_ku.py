"""Write N fakes from the clean generator (probe/ku_gen.py, search-best parameters) in the competition format.

    python probe/sim_series_ku.py N   # -> outputs/aug/series_simku.parquet, outputs/aug/meta_simku.parquet (label, family, kind)
"""
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, "probe")
from ku_gen import DEFAULT, gen

N = int(sys.argv[1])
b = {k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v) for k, v in json.load(open("outputs/factory/ku_gen_best.json")).items()}
P = dict(DEFAULT, **b)
out = Parallel(n_jobs=-1, batch_size=64)(delayed(gen)(P, 7 * 10 ** 6 + s, bool(s % 2)) for s in range(N))
rows, meta = [], []
for s, (x, n0, info) in enumerate(out):
    n = len(x)
    rows.append(pd.DataFrame(dict(id=f"sku{s:06d}", time=np.arange(n), value=x.astype(np.float64), period=(np.arange(n) >= n0).astype(np.int8))))
    meta.append(dict(id=f"sku{s:06d}", label=s % 2, family=info["family"], kind=info["kind"]))
pd.concat(rows, ignore_index=True).to_parquet("outputs/aug/series_simku.parquet")
pd.DataFrame(meta).set_index("id").to_parquet("outputs/aug/meta_simku.parquet")
print("simku series", N)
