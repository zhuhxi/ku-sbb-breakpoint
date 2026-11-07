"""Real break series with a large measured level jump (|level| > 1): is it a clean step? Plot 12 random ones, and
compare the shape statistics of the jump (immediate vs gradual, pre-boundary trend) with menu "L abc40 ns35".

    python probe/big_level.py
"""
import sys
import time

sys.argv = [sys.argv[0], "--menu", "L abc40 ns35", "--only-break", "1"]
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
F = pd.concat([pd.DataFrame(Parallel(n_jobs=30)(delayed(bc.feats)(v, n0) for v, n0 in L)).assign(src=k)
               for k, L in [("real", real), ("fake", fake)]], ignore_index=True)
F["series"] = real + fake
big = F[(F.level.abs() > 1) & F.cls.isin(["WN", "AR1", "AR2"])]
cols = ["local: first 30 after vs last 30 before", "gradual: level late vs early after", "trend slope", "robust level (median)", "n"]
print("median of each statistic among series with |level| > 1")
print(big.groupby("src")[cols].median().T.to_string(float_format="%.2f"))
print("count:", big.src.value_counts().to_dict())
# share whose jump is mostly gradual: late-after level differs from early-after by more than half the jump
big = big.assign(gradual=big["gradual: level late vs early after"] > 0.5 * big.level.abs())
print("share with a gradual (drift-like) jump:", big.groupby("src").gradual.mean().round(2).to_dict())
R = big[big.src == "real"].sample(12, random_state=0)
fig, ax = plt.subplots(4, 3, figsize=(15, 11))
for a, (_, row) in zip(ax.ravel(), R.iterrows()):
    v, n0 = row.series
    a.plot(v, lw=.6, color="0.4")
    a.plot(pd.Series(v).rolling(30, center=True).mean(), color="tab:blue", lw=1.2)
    a.hlines([v[:n0].mean(), v[n0:].mean()], [0, n0], [n0, len(v)], color="tab:red", lw=1.5)
    a.axvline(n0, color="k", ls="--", lw=.8)
    a.set_title(f"level {row.level:.2f}  {row.cls}  gradual {row.gradual}", fontsize=9)
fig.tight_layout()
fig.savefig("outputs/probe/big_level.png", dpi=80)
print(f"total {time.time() - t0:.0f}s")
