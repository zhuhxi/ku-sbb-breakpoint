"""SEGMENT SWAP view: series [post | pre] (each segment keeps its own time order), boundary at n - n0, label kept.
features.py on train -> outputs/swap_feat_train.parquet"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "."); from features import build_features
t = time.time(); S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"], kind="stable").reset_index(drop=True)
n0 = S.groupby("id")["period"].transform(lambda p: (p == 0).sum()); n = S.groupby("id")["time"].transform("size"); k = S.groupby("id").cumcount()
newt = np.where(S["period"] == 1, k - n0, k + (n - n0)); S = S.assign(time=newt, period=(1 - S["period"]).astype(S["period"].dtype)).sort_values(["id", "time"], kind="stable").reset_index(drop=True)
F = build_features(S); F.to_parquet("outputs/swap_feat_train.parquet"); print(F.shape, f"{time.time()-t:.0f}s")
