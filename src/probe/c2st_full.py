"""No-break C2ST with the full model features: real label-0 series vs generated no-break series (outputs/aug/X_sim.parquet,
probe/sim_series.py + aug_features.py sim; even sims are no-break), on the combo10g columns both have.
Prints the AUC and the columns the classifier leans on, with real / fake medians and single-column AUC.

    python probe/c2st_full.py
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
src = open("probe/param_quick.py").read()
exec(src[src.index("lab = pd.read_csv"):src.index("tr, va = next")])      # real X, y (combo10g columns)
S = pd.read_parquet("outputs/aug/X_sim.parquet")
M = pd.read_parquet("outputs/aug/meta_sim.parquet")
S = S.reindex(M.index[M.label == 0])
cols = [c for c in X.columns if c in S.columns and not ("--nobits" in ARGV and c.startswith("mb_"))]
R = X.loc[y == 0, cols].sample(len(S), random_state=0)
F = pd.concat([R, S[cols]]).replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
t = np.r_[np.zeros(len(R)), np.ones(len(S))]
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, verbose=-1, n_jobs=16)
p = cross_val_predict(lgb.LGBMClassifier(**P), F, t, cv=5, method="predict_proba")[:, 1]
imp = pd.Series(lgb.LGBMClassifier(**P).fit(F, t).booster_.feature_importance("gain"), cols)
print(f"{len(R)} real vs {len(S)} fake no-break, {len(cols)} columns: C2ST AUC {roc_auc_score(t, p):.3f}")
top = imp.sort_values(ascending=False).index[:15]
au = {c: roc_auc_score(t, F[c].fillna(F[c].median())) for c in top}
print(pd.DataFrame({"gain share": (imp[top] / imp.sum()).round(3), "AUC fake>real": pd.Series(au).round(3),
                    "real med": R[top].median().round(3), "fake med": S[top].median().round(3)}).to_string())
