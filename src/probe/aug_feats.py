"""features.py on transformed train series: MODE neg (x -> -x) or revneg (reversed and negated) -> outputs/{MODE}_feat_train.parquet"""
import sys, time, numpy as np, pandas as pd
from features import build_features
mode = sys.argv[1]; t = time.time(); S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"], kind="stable")
if "rev" in mode:
    n = S.groupby("id")["time"].transform("max"); S = S.assign(time=(n - S["time"]).astype(S["time"].dtype), period=(1 - S["period"]).astype(S["period"].dtype)).sort_values(["id", "time"], kind="stable")
if "neg" in mode: S = S.assign(value=-S["value"])
if mode.startswith("crop"):   # keep the last F of the pre segment and the first F of the post segment (boundary kept)
    f = float(mode[4:]) / 100
    g = S.groupby("id"); n0 = g["period"].transform(lambda v: (v == 0).sum()); n = g["time"].transform("size"); tt = S["time"]
    keep = ((S["period"] == 0) & (tt >= n0 - np.ceil(f * n0))) | ((S["period"] == 1) & (tt < n0 + np.ceil(f * (n - n0))))
    S = S[keep]
F = build_features(S.reset_index(drop=True)); F.to_parquet(f"outputs/{mode}_feat_train.parquet"); print(mode, F.shape, f"{time.time()-t:.0f}s")
