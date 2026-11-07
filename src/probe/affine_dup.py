"""Are some series affine (a*x+b) copies of others, in full or in the pre-boundary segment?
Z-score fixed windows (start of series; last 64 before boundary; first 64 after; diffs at start),
cosine nearest neighbour over train+test."""
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
W = 64
out = {}
for part in ["train", "test"]:
    d = pd.read_parquet(f"data/{part}_series.parquet").sort_values(["id", "time"])
    for sid, g in d.groupby("id", sort=False):
        v, p = g["value"].to_numpy(), g["period"].to_numpy()
        n0 = int((p == 0).sum())
        out[(part, sid)] = (v[:W], v[max(n0 - W, 0):n0], v[n0:n0 + W], np.diff(v[:W + 1]))
keys = list(out)
def z(a):
    a = a - a.mean(); return a / (np.linalg.norm(a) + 1e-12)
for j, name in enumerate(["start", "pre_end", "post_start", "dstart"]):
    kk = [k for k in keys if len(out[k][j]) == W]
    M = np.array([z(out[k][j]) for k in kk])
    dist, idx = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=28).fit(M).kneighbors(M)
    c = 1 - dist[:, 1]
    print(name, "n", len(M), "max corr", c.max().round(4), "n>0.99", (c > 0.99).sum(), "n>0.9", (c > 0.9).sum(),
          "q99.9", np.quantile(c, 0.999).round(3), flush=True)
    top = np.argsort(-c)[:5]
    print("   top pairs", [(kk[i], kk[idx[i, 1]], round(c[i], 4)) for i in top], flush=True)
