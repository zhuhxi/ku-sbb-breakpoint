"""Probe: single boundary step bits, and bit grid of the differences dx (hidden increment operands)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bits import bit_frame

def grid(v):
    b = np.abs(v).view(np.uint64)
    mant = (b & ((1 << 52) - 1)) | (1 << 52)
    low = mant & (~mant + np.uint64(1))
    e = ((b >> 52) & 0x7FF).astype(np.int64) - 1023
    L = e - 52 + np.log2(low.astype(np.float64)).astype(np.int64)
    return np.where(v == 0, 9999, L)

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
resc = pd.read_parquet("outputs/magic_b_train.parquet")["mb_rescaled"]
d = bit_frame("data/train_series.parquet")
d = d[~d["id"].isin(resc[resc == 1].index)]
v = d["value"].to_numpy(); prev = d.groupby("id")["value"].shift(1).to_numpy()
d["Lv"] = grid(v); d["Lp"] = grid(np.nan_to_num(prev))
dx = v - np.nan_to_num(prev)
d["Ldx"] = grid(dx)
d["ex"] = d["Ldx"] - np.maximum(d["Lv"], d["Lp"])     # extra zero bits of dx beyond what the two grids force
d["hold"] = (d["tz"] >= np.floor(d["lr"]) - 1).astype(float)
d["op"] = d["E"] - d["eprev"]
d = d[d["time"] > 0]
y = lab
def A(s):
    s = s.dropna(); yy = y.loc[s.index]; a = roc_auc_score(yy, s); return round(max(a, 1 - a), 4)
print("single steps around the boundary (non-rescaled):")
for r in (-2, -1, 0, 1, 2):
    x = d[d["rel"] == r].set_index("id")
    print(f"  rel {r:+d}: " + "  ".join(f"{c} {A(x[c])} (mean {x[c].mean():.3f})" for c in ("tz", "op", "hold", "ex", "lr")))
print("\nmean by label at rel 0:"); x = d[d["rel"] == 0].set_index("id"); print(x.groupby(y.loc[x.index].values)[["tz", "op", "hold", "ex"]].mean())
print("\ndx grid excess per segment:")
g = d.groupby(["id", "period"])["ex"].agg(["mean", lambda s: (s > 0).mean(), lambda s: s.clip(lower=0).quantile(0.9)]).unstack()
g.columns = [f"{a}_{b}" for a, b in g.columns]
for c in g: print(f"  {c}: AUC {A(g[c])}")
