"""Quora-style check: does the way samples / boundaries were chosen carry the label? (train labels only)
1) n, n0, n1 = n - n0, n0/n and residues n0 mod k: single AUCs + LightGBM 5-fold AUC on them;
2) is the boundary a special point? rates of: x[n0] or x[n0-1] local max/min within +-5, sign change of x - median
   across the boundary, label 1 vs label 0 (and at a random position as control).

    python probe/len_leak.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
S["id"] = S["id"].astype(str)
rows, rng = [], np.random.default_rng(0)
for k, g in S.groupby("id", sort=False):
    x = g["value"].to_numpy(np.float64)
    n, n0 = len(x), int((g["period"] == 0).sum())
    def special(c):
        w = x[c - 5:c + 6]
        med = np.median(x)
        return (x[c] in (w.max(), w.min())) or (x[c - 1] in (w.max(), w.min())), (x[c - 1] - med) * (x[c] - med) < 0
    c = int(rng.integers(20, n - 20))
    (e0, s0), (e1, s1) = special(n0), special(c)
    rows.append(dict(id=k, n=n, n0=n0, n1=n - n0, frac=n0 / n, m2=n0 % 2, m5=n0 % 5, m10=n0 % 10, m8=n0 % 8, nm10=n % 10,
                     ext=e0, sgn=s0, ext_rand=e1, sgn_rand=s1))
D = pd.DataFrame(rows).set_index("id")
y = L.loc[D.index].to_numpy()
for c in ("n", "n0", "n1", "frac"):
    a = roc_auc_score(y, D[c])
    print(f"{c:5s} AUC {max(a, 1 - a):.4f}")
cols = ["n", "n0", "n1", "frac", "m2", "m5", "m10", "m8", "nm10"]
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, verbose=-1), D[cols], y, cv=5, method="predict_proba")[:, 1]
print(f"LightGBM on lengths / positions / residues: 5-fold AUC {roc_auc_score(y, p):.4f}")
for c in ("ext", "sgn"):
    print(f"{c}: label1 {D.loc[y == 1, c].mean():.4f}  label0 {D.loc[y == 0, c].mean():.4f}  | random pos: label1 {D.loc[y == 1, c + '_rand'].mean():.4f} label0 {D.loc[y == 0, c + '_rand'].mean():.4f}")
