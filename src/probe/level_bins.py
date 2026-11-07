"""Measured statistic (default |level|) (probe/hidden_diag.py) per size bin: real break series vs a menu; all series, then short and long ones separately.

    python probe/level_bins.py ["noise shape (kurtosis)" MENU]

    python probe/level_bins.py
"""
import sys
import time

COL = sys.argv[1] if len(sys.argv) > 1 else "level"
MENU_NAME = sys.argv[2] if len(sys.argv) > 2 else "L tuned"
sys.argv = [sys.argv[0], "--menu", MENU_NAME, "--only-break", "1"]
import importlib.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

s = importlib.util.spec_from_file_location("bc", "probe/break_c2st.py")
bc = importlib.util.module_from_spec(s); s.loader.exec_module(bc)

t0 = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = set(np.random.default_rng(7).choice(lab.index[lab == 1], 1500, False))
real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
fake = Parallel(n_jobs=30)(delayed(bc.cb.gen)(bc.MENU, 5 * 10 ** 6 + 10 ** 5 + i, True, True) for i in range(1500))
F = pd.DataFrame(Parallel(n_jobs=30)(delayed(bc.feats)(v, n0) for v, n0 in real + fake))
F["src"] = ["real"] * len(real) + ["fake"] * len(fake)
F = F[F.cls.isin(["WN", "AR1", "AR2"])]
bins = {"level": [0, .1, .2, .3, .5, .75, 1, 1.5, 2, 3, 99], "n": [256, 400, 550, 700, 850, 1025],
        "frac": [.3, .4, .5, .6, .7]}.get(COL, [0, .05, .1, .2, .3, .5, .75, 1, 2, 99])
F["bin"] = pd.cut(F[COL].abs(), bins, right=False)
for name, part in [("all", F), ("short (n < 640)", F[F.n < 640]), ("long (n >= 640)", F[F.n >= 640])]:
    T = pd.crosstab(part.bin, part.src, normalize="columns")[["real", "fake"]] * 100
    T["fake/real"] = T.fake / T.real
    print(f"\n{name}")
    print(T.to_string(float_format="%.2f"))
print(f"total {time.time() - t0:.1f}s")
