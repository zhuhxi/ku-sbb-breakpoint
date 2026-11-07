"""Santander-style check: do series share values or pieces? (train + test, series only, no labels)
1) exact float64 values that occur in more than one series;
2) shared pieces invariant to the outer shell (x = base + size * core): windows of 3 consecutive ratios d_t / d_{t-1}
   of first differences, rounded to 1e-9, hashed; windows found in more than one series.

    python probe/dup_check.py
"""
import time
import numpy as np
import pandas as pd

t0 = time.time()
S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
v = S["value"].to_numpy(np.float64)
sid = pd.factorize(S["id"])[0]
u, inv = np.unique(v, return_inverse=True)
pairs = pd.DataFrame({"u": inv, "s": sid}).drop_duplicates()
ns = np.bincount(pairs.u, minlength=len(u))
print(f"{len(v)} values, {len(u)} distinct; values in >1 series: {(ns > 1).sum()} "
      f"(covering {np.isin(inv, np.flatnonzero(ns > 1)).mean():.4%} of points); top repeated: {u[np.argsort(-ns)[:5]].round(6).tolist()} x{np.sort(ns)[::-1][:5].tolist()}")
d = np.diff(v)
same = sid[1:] == sid[:-1]
r = np.where(same[1:] & same[:-1], d[1:] / np.where(d[:-1] == 0, np.nan, d[:-1]), np.nan)
r = np.round(r, 9)
w = np.column_stack([r[:-2], r[1:-1], r[2:]])
ok = ~np.isnan(w).any(1) & (sid[3:-0 or None][:len(w)] == sid[:len(w)])
key = pd.util.hash_pandas_object(pd.DataFrame(w[ok]), index=False).to_numpy()
ws = sid[:len(w)][ok]
P = pd.DataFrame({"k": key, "s": ws}).drop_duplicates()
c = P.k.value_counts()
multi = c[c > 1]
print(f"shell-invariant 3-ratio windows: {len(key)}, shared by >1 series: {len(multi)}; series involved: "
      f"{P[P.k.isin(multi.index)].s.nunique()}  ({time.time() - t0:.1f}s)")
