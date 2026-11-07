"""Where does the current best miss breaks? Best = equal rank blend of combo10gnj LGB + interaction arms 1, 2 (OOF).
Groups: missed = label 1 scored below the median of label-0 scores; caught = label 1 in the top half of label-1 scores;
neg = label 0. Per fitted pre-boundary structure (genA g_struct): share of each group, miss rate of breaks.
Then: which simple descriptors separate missed breaks from negatives (single AUC)?

    python probe/miss_scan.py
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

STRUCT = ["WN", "AR1", "AR2", "MA1", "ARMA11", "RW", "SIN+WN", "SIN+AR1"]
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = L.to_numpy()
r = lambda v: rankdata(v) / len(v)
I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
s = (r(np.load("outputs/factory/confirm_combo10gnj.npz")["oof"]) + r(I[1]) + r(I[2])) / 3
grp = np.where(y == 0, "neg", np.where(s < np.median(s[y == 0]), "missed", np.where(s > np.median(s[y == 1]), "caught", "mid")))
A = pd.read_parquet("outputs/factory/genA.parquet")
A.index = A.index.astype(str)
A = A.reindex(L.index)
B = pd.read_parquet("outputs/factory/genB.parquet")
B.index = B.index.astype(str)
F = pd.read_parquet("outputs/feat_train.parquet")
F.index = F.index.astype(str)
D = pd.concat([A, B.reindex(L.index), F.reindex(L.index)], axis=1)
D = D.loc[:, ~D.columns.duplicated()]
D["grp"], D["st"] = grp, A["g_struct"].map(dict(enumerate(STRUCT)))
print("share of breaks missed:", round((grp == "missed").sum() / (y == 1).sum(), 3))
T = pd.crosstab(D.st, D.grp)
T["miss rate of breaks"] = (T["missed"] / (T["missed"] + T["mid"] + T["caught"])).round(3)
T["share of all missed"] = (T["missed"] / T["missed"].sum()).round(3)
print(T[["neg", "missed", "caught", "miss rate of breaks", "share of all missed"]].to_string())
m = D[D.grp.isin(["missed", "neg"])]
t = (m.grp == "missed").to_numpy()
au = {}
for c in m.columns.drop(["grp", "st"]):
    v = pd.to_numeric(m[c], errors="coerce")
    if v.notna().mean() > .9 and v.nunique() > 5:
        a = roc_auc_score(t, v.fillna(v.median()))
        au[c] = max(a, 1 - a)
print("descriptors separating missed breaks from negatives (top 10 single AUC):")
print(pd.Series(au).sort_values(ascending=False).head(10).round(3).to_string())
