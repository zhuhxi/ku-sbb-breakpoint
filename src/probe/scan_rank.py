"""Is the boundary special inside its own series? Scan cuts every 5 points; at each cut compare the w points left vs
right (w = min(n0, n1, 100)): |log sd ratio| and |mean diff| / pooled sd. Feature = share of cuts (at least w from the
ends, excluding +-w/2 around the boundary) whose value is below the boundary's value (1 = boundary is the biggest).
Checked on the confusable set: short series (n < 500) whose combo10g score is in the top 30%.

    python probe/scan_rank.py
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

t = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
g = pd.Series(np.load("outputs/factory/confirm_combo10g.npz")["oof"], index=lab.index)
F = pd.read_parquet("outputs/feat_train.parquet").reindex(lab.index)
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
L = S.groupby("id").size().reindex(lab.index)
ids = lab.index[(L < 500).to_numpy()]


def one(x, n0):
    w = max(20, min(n0, len(x) - n0, 100))
    cs, cs2 = np.r_[0, np.cumsum(x)], np.r_[0, np.cumsum(x * x)]
    def side(i, j):
        m = (cs[j] - cs[i]) / (j - i)
        return m, np.sqrt(max((cs2[j] - cs2[i]) / (j - i) - m * m, 1e-12))
    def stat(c):
        m0, s0 = side(c - w, c)
        m1, s1 = side(c, c + w)
        return abs(np.log(s1 / s0)), abs(m1 - m0) / np.sqrt((s0 ** 2 + s1 ** 2) / 2)
    b = stat(n0) if n0 >= w and len(x) - n0 >= w else (np.nan, np.nan)
    cuts = [c for c in range(w, len(x) - w + 1, 5) if abs(c - n0) > w // 2]
    if len(cuts) < 5:
        return np.nan, np.nan
    v = np.array([stat(c) for c in cuts])
    return (v[:, 0] < b[0]).mean(), (v[:, 1] < b[1]).mean()


G = {k: (d["value"].to_numpy(), int((d["period"] == 0).sum())) for k, d in S[S.id.isin(ids)].groupby("id")}
R = pd.DataFrame(Parallel(n_jobs=30)(delayed(one)(*G[k]) for k in ids), index=ids, columns=["scan_sd", "scan_mean"])
print(f"{len(ids)} short series scanned, {time.time() - t:.1f}s")
conf = g[ids] >= g.quantile(0.7)
y = lab[ids][conf]
print(f"confusable set: {conf.sum()} series, {y.mean():.0%} breaks;  combo10g AUC inside it {roc_auc_score(y, g[ids][conf]):.3f}")
for nm, v in (("old vshift_vs_drift", F.loc[ids, "vshift_vs_drift"]), ("old shift_vs_drift", F.loc[ids, "shift_vs_drift"]),
              ("new scan_sd", R["scan_sd"]), ("new scan_mean", R["scan_mean"])):
    v = v[conf]
    ok = v.notna()
    print(f"  {nm:20s} AUC inside confusable set {roc_auc_score(y[ok], v[ok]):.3f}  (nan {1 - ok.mean():.0%})")
print(f"{time.time() - t:.1f}s")
