"""KU-SBB bit-level "magic" columns (set B) for train and test.

    python brute/magic_b.py

Per series: precision grid L = exponent - 52 + trailing zero bits (Lmin / Lmed / Lq90 before,
after, after-before), LSB parity (before, after, diff), and the rescaled-family marker
(mean trailing zeros where |v_{t-1}|/|v_t| > 8; tzc < 1.5 = final values were rescaled).
In rescaled series the low bits are random, so every bit column except the marker is NaN there.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bits import bit_frame  # noqa: E402


def magic_b(path):
    d = bit_frame(path)
    d["L"] = d["exp"] - 52 + d["tz"]
    d["par"] = (d["value"].to_numpy().view(np.uint64) & 1).astype(np.int8)
    g = d.groupby(["id", "period"])
    q90 = lambda s: s.quantile(0.9)
    G = g.agg(Lmin=("L", "min"), Lmed=("L", "median"), Lq90=("L", q90), par=("par", "mean")).unstack()
    out = pd.DataFrame(index=G.index)
    for k in ("Lmin", "Lmed", "Lq90", "par"):
        out[f"{k}_b"], out[f"{k}_a"] = G[k][0], G[k][1]
        out[f"{k}_d"] = G[k][1] - G[k][0]
    c = d[d["lr"] > 3]
    T = c.groupby(["id", "period"])["tz"].mean().unstack()
    out["tzc_b"], out["tzc_a"] = T[0].reindex(out.index), T[1].reindex(out.index)
    out["tzc"] = c.groupby("id")["tz"].mean().reindex(out.index)
    out["rescaled"] = (out["tzc"] < 1.5).astype(float)
    bitcols = [x for x in out.columns if x not in ("tzc", "rescaled")]
    out.loc[out["rescaled"] == 1, bitcols] = np.nan
    return out.add_prefix("mb_")


if __name__ == "__main__":
    for split in ("train", "test"):
        B = magic_b(f"data/{split}_series.parquet")
        B.to_parquet(f"outputs/magic_b_{split}.parquet")
        print(split, B.shape, "rescaled share", round(B["mb_rescaled"].mean(), 4))
