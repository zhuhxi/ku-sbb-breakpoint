"""Per-value float bit table for KU-SBB series (shared by the forensics experiments)."""
import numpy as np
import pandas as pd


def bit_frame(path):
    d = pd.read_parquet(path)
    v = d["value"].to_numpy()
    b = v.view(np.uint64)
    mant = (b & ((1 << 52) - 1)) | (1 << 52)
    low = mant & (~mant + np.uint64(1))
    d["exp"] = ((b >> 52) & 0x7FF).astype(np.int16) - 1023
    d["tz"] = np.log2(low.astype(np.float64)).astype(np.int16)   # trailing zero bits of the mantissa
    d["E"] = d["exp"] + d["tz"]                                  # exponent of the largest operand (lower bound)
    prev = d.groupby("id")["value"].shift(1).to_numpy()
    d["lr"] = np.log2(np.abs(prev) / (np.abs(v) + 1e-300))       # log2 |v_{t-1}| / |v_t|
    d["eprev"] = np.floor(np.log2(np.abs(prev) + 1e-300))
    d["rel"] = d["time"] - d.groupby("id")["period"].transform(lambda p: int((p == 0).sum()))
    return d
