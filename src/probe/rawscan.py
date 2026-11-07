"""Broad raw-representation scan: per segment histograms of float properties -> can LightGBM predict the label?"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
d = pd.read_parquet("data/train_series.parquet")
v = d["value"].to_numpy(); b = v.view(np.uint64)
mant = (b & ((1 << 52) - 1)) | (1 << 52)
d["tz"] = np.minimum(np.log2((mant & (~mant + np.uint64(1))).astype(float)).astype(int), 12)
d["rlen"] = np.minimum(np.fromiter((len(repr(abs(x))) for x in v), int, len(v)), 20)   # shortest decimal repr length
d["lowbyte"] = (b & 0xF).astype(int)                                                      # lowest 4 mantissa bits
d["frac_digit"] = np.fromiter((repr(x)[-1] for x in v), "U1", len(v))
d["sign"] = (v < 0).astype(int)
parts = []
for c in ("tz", "rlen", "lowbyte", "frac_digit"):
    h = pd.crosstab([d["id"], d["period"]], d[c], normalize="index").unstack()
    h.columns = [f"{c}_{a}_p{p}" for a, p in h.columns]
    parts.append(h)
X = pd.concat(parts, axis=1).reindex(lab.index).fillna(0)
for p in ("tz", "rlen", "lowbyte", "frac"):     # after minus before
    for c in [c for c in X if c.startswith(p) and c.endswith("_p1")]:
        X[c[:-3] + "_d"] = X[c] - X[c[:-3] + "_p0"]
y = lab.values
X.to_parquet("outputs/rawscan_train.parquet")
import sys
if len(sys.argv) > 1: sys.exit()
m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, colsample_bytree=0.5, verbose=-1, n_jobs=24)
o = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
print(X.shape, "label AUC of raw-representation histograms:", round(roc_auc_score(y, o), 4))
for g in ("tz", "rlen", "lowbyte", "frac"):
    cols = [c for c in X if c.startswith(g)]
    o = cross_val_predict(m, X[cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    print(f"  {g:8s} {len(cols):3d} cols  AUC {roc_auc_score(y, o):.4f}")
