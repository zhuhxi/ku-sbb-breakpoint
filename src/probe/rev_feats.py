"""Time-reversal augmentation probe: build features.py features on REVERSED train series (period flipped so the boundary is
at n - n0), -> outputs/rev_feat_train.parquet. Same ids (suffix-free) as train."""
import time, pandas as pd, numpy as np
from features import build_features
t = time.time(); S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"], kind="stable")
n = S.groupby("id")["time"].transform("max")
R = S.assign(time=(n - S["time"]).astype(S["time"].dtype), period=(1 - S["period"]).astype(S["period"].dtype)).sort_values(["id", "time"], kind="stable").reset_index(drop=True)
F = build_features(R); F.to_parquet("outputs/rev_feat_train.parquet"); print(F.shape, f"{time.time()-t:.0f}s")
