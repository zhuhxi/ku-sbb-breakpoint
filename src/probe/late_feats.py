"""features.py on series whose post segment starts LATER: the first FRAC of the post-boundary points is dropped (tests lasting vs
transient change). -> outputs/factory/latefe.parquet (prefix l_). REVL=path: same on time-reversed series."""
import os, sys, time, pandas as pd, numpy as np
sys.path.insert(0, "."); from features import build_features
FR = float(os.environ.get("FRAC", .2)); t = time.time(); out = []
for part in ("train", "test"):
    S = pd.read_parquet(f"data/{part}_series.parquet").sort_values(["id", "time"], kind="stable").reset_index(drop=True)
    if os.environ.get("REVL"):
        n = S.groupby("id")["time"].transform("max")
        S = S.assign(time=(n - S["time"]).astype(S["time"].dtype), period=(1 - S["period"]).astype(S["period"].dtype)).sort_values(["id", "time"], kind="stable").reset_index(drop=True)
    k = S[S.period == 1].groupby("id").cumcount(); npost = S[S.period == 1].groupby("id")["time"].transform("size")
    drop = k < (FR * npost).astype(int); S = S.drop(index=drop[drop].index).reset_index(drop=True)
    out.append(build_features(S)); print(part, out[-1].shape, f"{time.time()-t:.0f}s", flush=True)
F = pd.concat(out); F.columns = ["l_" + c for c in F.columns]; F.index.name = "id"; F.to_parquet(os.environ.get("REVL", "outputs/factory/latefe.parquet")); print(F.shape)
