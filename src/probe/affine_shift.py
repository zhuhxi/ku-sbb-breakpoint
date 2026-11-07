"""Time-shifted affine copies: if x' = a*x + b (any offset), then r_t = dx_t / dx_{t-1} is identical.
Exact-match triples of consecutive ratios (rounded to 1e-6) across every position of every series."""
import numpy as np, pandas as pd
parts = []
for part in ["train", "test"]:
    d = pd.read_parquet(f"data/{part}_series.parquet").sort_values(["id", "time"])
    v = d["value"].to_numpy(); ids = d["id"].to_numpy()
    dx = np.diff(v); same = ids[1:] == ids[:-1]
    r = dx[1:] / np.where(dx[:-1] == 0, np.nan, dx[:-1]); ok = same[1:] & same[:-1]
    r = np.where(ok, r, np.nan)
    k = np.c_[r[:-2], r[1:-1], r[2:]]
    good = np.isfinite(k).all(1) & (np.abs(k) < 1e6).all(1)
    k = np.round(k[good], 2)
    parts.append(pd.DataFrame({"a": k[:, 0], "b": k[:, 1], "c": k[:, 2], "id": ids[2:-2][good], "part": part}))
D = pd.concat(parts, ignore_index=True)
print("triples", len(D), flush=True)
g = D.groupby(["a", "b", "c"])["id"].nunique()
hits = g[g > 1]
print("triples shared by >1 series:", len(hits))
if len(hits):
    m = D.merge(hits.rename("n").reset_index(), on=["a", "b", "c"])
    pairs = m.groupby(["a", "b", "c"])["id"].apply(lambda s: tuple(sorted(set(s))))
    vc = pairs[pairs.map(len) == 2].value_counts(); print("pairs sharing >=3 triples:", (vc >= 3).sum()); print(vc.head(10))
