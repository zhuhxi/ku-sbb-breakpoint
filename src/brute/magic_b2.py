"""KU-SBB detailed bit-profile features (set B2) for train and test.

    python brute/magic_b2.py

The generator builds v_t from v_{t-1}; the low bits of v_t record the size of the operands of
that computation. Per series and segment (b = before, a = after, d = after - before):
  * tz_lr{k}: mean trailing zero bits for values whose |v_{t-1}|/|v_t| lies in log2-bin k;
  * gap / hold: how closely the cancellation rule tz ~ lr holds where lr > 2;
  * op_med / op_q25: operand exponent relative to the previous value (E - exp(v_{t-1}));
  * par_lo / par_mid / par_hi: LSB parity by magnitude band (exponent < -2, -2..0, > 0);
  * seam_*: the same rule quantities over the first 20 after-values vs the last 20 before-values.
Rescaled series (random low bits) are NaN everywhere.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bits import bit_frame  # noqa: E402

LR_BINS = [-np.inf, 0, 1, 2, 3, 4, 6, np.inf]


def magic_b2(path, rescaled):
    d = bit_frame(path)
    d["par"] = (d["value"].to_numpy().view(np.uint64) & 1).astype(np.int8)
    d["gap"] = d["tz"] - d["lr"]
    d["op"] = d["E"] - d["eprev"]
    d["hold"] = (d["tz"] >= np.floor(d["lr"]) - 1).astype(np.float32)
    d["lrb"] = pd.cut(d["lr"], LR_BINS, labels=False)
    d["band"] = np.select([d["exp"] < -2, d["exp"] <= 0], [0, 1], 2)
    seg = {"b": d["period"] == 0, "a": d["period"] == 1}
    parts = []
    for s, m in seg.items():
        x = d[m]
        t = x.groupby(["id", "lrb"])["tz"].mean().unstack()
        t.columns = [f"tz_lr{int(k)}_{s}" for k in t.columns]
        c = x[x["lr"] > 2].groupby("id")
        o = pd.DataFrame({f"gap_{s}": c["gap"].mean(), f"hold_{s}": c["hold"].mean(),
                          f"op_med_{s}": c["op"].median(), f"op_q25_{s}": c["op"].quantile(0.25)})
        p = x.groupby(["id", "band"])["par"].mean().unstack()
        p.columns = [f"par_{['lo', 'mid', 'hi'][int(k)]}_{s}" for k in p.columns]
        parts += [t, o, p]
    near_a = d[(d["rel"] >= 0) & (d["rel"] < 20) & (d["lr"] > 2)].groupby("id")
    near_b = d[(d["rel"] < 0) & (d["rel"] >= -20) & (d["lr"] > 2)].groupby("id")
    parts.append(pd.DataFrame({"seam_hold_a": near_a["hold"].mean(), "seam_hold_b": near_b["hold"].mean(),
                               "seam_gap_a": near_a["gap"].mean(), "seam_gap_b": near_b["gap"].mean()}))
    out = pd.concat(parts, axis=1)
    for c in [c for c in out.columns if c.endswith("_a")]:
        b = c[:-2] + "_b"
        if b in out:
            out[c[:-2] + "_d"] = out[c] - out[b]
    out.loc[out.index.isin(rescaled[rescaled == 1].index)] = np.nan
    return out.add_prefix("b2_").astype(np.float32)


if __name__ == "__main__":
    for split in ("train", "test"):
        resc = pd.read_parquet(f"outputs/magic_b_{split}.parquet")["mb_rescaled"]
        B2 = magic_b2(f"data/{split}_series.parquet", resc)
        B2.to_parquet(f"outputs/magic_b2_{split}.parquet")
        print(split, B2.shape)
