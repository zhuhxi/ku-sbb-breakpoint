"""Do candidate features explain the model's errors? Inside a subset, predict the label from the combo10g OOF score
alone vs score + candidate columns (small LightGBM, repeated 5-fold CV). A clear gain means the candidates carry
information the model is missing there; the top columns say which.

    python probe/err_bank.py PARQUET PATTERN     # e.g. outputs/harvest_train.parquet '^H_c22_'
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

t = time.time()
path, pat = sys.argv[1], sys.argv[2]
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
g = pd.Series(np.load("outputs/factory/confirm_combo10g.npz")["oof"], index=lab.index)
C = pd.read_parquet(path).reindex(lab.index)
C = C.loc[:, C.columns.str.contains(pat, regex=True)].replace([np.inf, -np.inf], np.nan)
L = pd.read_parquet("data/train_series.parquet", columns=["id"]).groupby("id").size().reindex(lab.index)
sets = {"short, top-30% score": (L < 500) & (g >= g.quantile(.7)), "all short": L < 500}
m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=30, colsample_bytree=0.7, verbose=-1, n_jobs=8)
print(f"{C.shape[1]} candidate columns")
for nm, s in sets.items():
    y = lab[s].to_numpy()
    base = roc_auc_score(y, g[s])
    out = []
    for r in range(3):
        cv = StratifiedKFold(5, shuffle=True, random_state=r)
        p0 = cross_val_predict(m, g[s].to_frame("g"), y, cv=cv, method="predict_proba")[:, 1]
        p1 = cross_val_predict(m, pd.concat([g[s].rename("g"), C[s]], axis=1), y, cv=cv, method="predict_proba")[:, 1]
        out.append((roc_auc_score(y, p0), roc_auc_score(y, p1)))
    a0, a1 = np.mean(out, 0)
    print(f"{nm:22s} n={s.sum()} breaks {y.mean():.0%}: score raw {base:.4f} | score-only model {a0:.4f} -> +candidates {a1:.4f} ({a1 - a0:+.4f}, per-repeat {[round(b - a, 4) for a, b in out]})")
    if nm == "all short":
        imp = pd.Series(m.fit(pd.concat([g[s].rename("g"), C[s]], axis=1), y).booster_.feature_importance("gain"), index=["g"] + list(C.columns))
        print("  top candidates:", (imp.drop("g") / imp.sum()).sort_values(ascending=False).head(5).round(3).to_dict())
print(f"{time.time() - t:.1f}s")
