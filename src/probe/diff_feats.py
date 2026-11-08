"""features.py on FIRST DIFFERENCES of each series (helps random-walk series): -> outputs/factory/difffe.parquet (train+test, prefix d_)."""
import os, sys, time, pandas as pd, numpy as np
sys.path.insert(0, "."); from features import build_features
t = time.time(); out = []
for part in ("train", "test"):
    S = pd.read_parquet(f"data/{part}_series.parquet").sort_values(["id", "time"], kind="stable")
    if os.environ.get("REVD"):   # time-reversed series (boundary at n - n0), as in probe/rev_feats.py
        n = S.groupby("id")["time"].transform("max")
        S = S.assign(time=(n - S["time"]).astype(S["time"].dtype), period=(1 - S["period"]).astype(S["period"].dtype)).sort_values(["id", "time"], kind="stable").reset_index(drop=True)
    S = S.assign(value=S.groupby("id")["value"].diff()).dropna(subset=["value"]).reset_index(drop=True)
    out.append(build_features(S)); print(part, out[-1].shape, f"{time.time()-t:.0f}s", flush=True)
F = pd.concat(out); F.columns = ["d_" + c for c in F.columns]; F.index.name = "id"; F.to_parquet(os.environ.get("REVD", "outputs/factory/difffe.parquet")); print(F.shape)
