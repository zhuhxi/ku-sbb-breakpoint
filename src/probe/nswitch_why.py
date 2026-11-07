"""Why does "a break also redraws the noise type" look less real? Before-vs-after statistics per bin for real break
series, menu "L abc40" without and with the noise switch (35%).

    python probe/nswitch_why.py
"""
import sys
import time

sys.argv = [sys.argv[0], "--menu", "L abc40", "--only-break", "1"]
import importlib.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

s = importlib.util.spec_from_file_location("bc", "probe/break_c2st.py")
bc = importlib.util.module_from_spec(s); s.loader.exec_module(bc)
base = bc.cb.BREAKS["L abc40"]
t0 = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = set(np.random.default_rng(7).choice(lab.index[lab == 1], 1500, False))
real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
D = {"real": real}
for q in (0, .35):
    D[f"switch {q:.0%}"] = Parallel(n_jobs=30)(delayed(bc.cb.gen)(dict(base, noise_switch=q), 5 * 10 ** 6 + 10 ** 5 + i, True, True) for i in range(1500))
F = pd.concat([pd.DataFrame(Parallel(n_jobs=30)(delayed(bc.feats)(v, n0) for v, n0 in L)).assign(src=k) for k, L in D.items()])
F = F[F.cls.isin(["WN", "AR1", "AR2"])]
for col, bins in [("spread", [0, .05, .1, .2, .35, .5, .75, 9]), ("robust spread (MAD)", [0, .05, .1, .2, .35, .5, .75, 9]),
                  ("noise shape (kurtosis)", [0, .1, .3, .5, 1, 2, 99])]:
    T = pd.crosstab(pd.cut(F[col].abs(), bins, right=False), F.src, normalize="columns")[list(D)] * 100
    print(f"\n{col}  (% of series per bin)")
    print(T.to_string(float_format="%.1f"))
print(f"total {time.time() - t0:.0f}s")
