"""Training-data augmentation for the single LGB (the model stays the same; only the training rows change).

    python augment.py            # writes outputs/aug/series_crop.parquet, series_spike.parquet, meta.parquet
    python augment.py --seed 2027 --suffix 2     # extra crops only: series_crop2.parquet, meta2.parquet

Error analysis (probe/hard_cases.py): the model is weakest on short segments (AUC 0.79) and on series with an
isolated spike after the boundary (0.80 vs 0.846). Two augmentations, one copy per training series each:

  crop  : keep the last n0' points of the before segment and the first n1' points of the after segment.
          Total length ~ U[256, n], boundary share in [0.3, 0.7] (the same limits as train and test). Label kept.
  spike : 1-2 single-point spikes of 6-12 robust sd (sign random) at random positions of the after segment, for
          BOTH labels, so a spike after the boundary carries no label information. Label kept.

Row ids are "<parent>_c" / "<parent>_s"; meta.parquet holds parent, kind and label. The magic B bit features of
spike rows must be copied from the parent (made-up spike values would carry their own bit signature).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--suffix", default="")
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
OUT = Path("outputs/aug")
OUT.mkdir(parents=True, exist_ok=True)
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"], kind="stable")


def crop(v, n0):
    n1 = len(v) - n0
    for _ in range(100):
        m = int(rng.integers(256, len(v) + 1))
        lo, hi = max(0.3, 1 - n1 / m), min(0.7, n0 / m)
        if lo <= hi:
            k0 = int(round(rng.uniform(lo, hi) * m))
            k0 = min(max(k0, m - n1), n0)
            return v[n0 - k0:n0 + (m - k0)], k0
    return v, n0


def spike(v, n0):
    v = v.copy()
    r = v - pd.Series(v).rolling(9, center=True, min_periods=1).median().to_numpy()
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
    for t in rng.choice(np.arange(n0, len(v)), size=int(rng.integers(1, 3)), replace=False):
        v[t] += rng.choice([-1, 1]) * rng.uniform(6, 12) * s
    return v


def frame(sid, v, n0):
    return pd.DataFrame({"id": sid, "time": np.arange(len(v)), "value": v, "period": (np.arange(len(v)) >= n0).astype(np.int8)})


C, P, meta = [], [], []
for sid, g in S.groupby("id", sort=False):
    v, n0 = g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())
    vc, k0 = crop(v, n0)
    sx = args.suffix
    C.append(frame(sid + "_c" + sx, vc, k0))
    meta.append((sid + "_c" + sx, sid, "crop" + sx, lab[sid]))
    if not sx:
        P.append(frame(sid + "_s", spike(v, n0), n0))
        meta.append((sid + "_s", sid, "spike", lab[sid]))
pd.concat(C).to_parquet(OUT / f"series_crop{args.suffix}.parquet")
if P:
    pd.concat(P).to_parquet(OUT / "series_spike.parquet")
M = pd.DataFrame(meta, columns=["id", "parent", "kind", "label"]).set_index("id")
M.to_parquet(OUT / f"meta{args.suffix}.parquet")
print(M.groupby("kind").size().to_dict(), "crop length", pd.concat(C).groupby("id").size().describe()[["min", "50%", "max"]].to_dict())
