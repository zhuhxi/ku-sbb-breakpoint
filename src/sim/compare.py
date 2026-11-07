"""Quantiles of chosen features, real vs simulated no-break series:  python sim/compare.py col1 col2 ..."""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "sim")
import numpy as np, pandas as pd
import simlib
from features import build_features
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
d = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
ids0 = lab.index[lab == 0][:4000]
G = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in d[d.id.isin(set(ids0))].groupby("id", sort=False)]
df, y, _ = simlib.simulate(simlib.make_donors(G), 3000, simlib.KNOBS, seed=3)
S = build_features(df); S = S[y.loc[S.index] == 0]
R = pd.read_parquet("outputs/feat_train.parquet").loc[ids0]
q = [0.05, 0.25, 0.5, 0.75, 0.95]
for c in sys.argv[1:]:
    print(f"{c:12s} real", R[c].quantile(q).round(3).tolist(), "\n             sim ", S[c].quantile(q).round(3).tolist())
