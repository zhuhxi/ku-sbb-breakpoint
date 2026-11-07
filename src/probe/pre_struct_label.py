"""Label rate by structure fitted on the PRE-boundary part only (not touched by the break).

    python probe/pre_struct_label.py [N]
"""
import importlib.util
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")
t0 = time.time()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("gf", "probe/gen_feats.py")
gf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gf)
y = pd.read_csv("data/train_labels.csv").set_index("id")["label"]
ids = set(np.random.default_rng(0).choice(y.index, N, replace=False))
S = pd.read_parquet("data/train_series.parquet")
S = S[S.id.isin(ids)].sort_values(["id", "time"])
items = [(i, g.value.to_numpy(np.float64)[g.period.to_numpy() == 0]) for i, g in S.groupby("id", sort=False)]
win = Parallel(n_jobs=-1, batch_size=16)(delayed(lambda a: gf.mle.fit("b", a)[1])(a) for _, a in items)
D = pd.DataFrame(dict(st=win, label=y.reindex([i for i, _ in items]).to_numpy()))
t = pd.crosstab(D.st, D.label)
print(f"pre-part structure, {len(D)} series: chi2 p {chi2_contingency(t)[1]:.3g}")
for k, r in t.iterrows():
    print(f"   {k:8s} n {r.sum():5d}  break share {r[1] / r.sum():.3f}  (+-{1.96 * np.sqrt(.25 / r.sum()):.3f})")
print(f"{time.time() - t0:.1f}s")
