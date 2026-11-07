"""TabPFN 3.5 the way the docs / papers recommend, fold 0 only (same StratifiedKFold(5, shuffle, 42) as the main runs):
  - whole fold-0 train part as context, each estimator sees SUBSAMPLE_SAMPLES rows (stratified), n_estimators of them
    (PriorLabs memory-optimisation cookbook);
  - the 888 columns ranked by LightGBM gain and dealt round-robin into G groups, one TabPFN per group, predictions
    averaged (feature-subset ensembling, "A Closer Look at TabPFN v2");
  - use 1: equal rank blend with the combo10gnj LGB OOF; use 2: TabPFN as an extra input next to the LGB OOF logit
    (stacking, ADIA 2nd place), checked by a 5-fold logistic regression inside the fold-0 validation part.

    TABPFN_TOKEN=... python probe/tabpfn_expert.py [G] [SUB] [NEST]   # SUB 0 = no row subsampling (all rows, docs limit 50k)
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from tabpfn import TabPFNClassifier

G = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SUB = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
NEST = int(sys.argv[3]) if len(sys.argv) > 3 else 4
sys.argv = sys.argv[:1]
t0 = time.time()
src = open("probe/param_quick.py").read()
exec(src[src.index("lab = pd.read_csv"):src.index("tr, va = next")])      # X (combo10g), y, ld
X = pd.concat([X, ld("outputs/factory/gnoise.parquet"), ld("outputs/factory/gjlr.parquet")], axis=1)
X = X.loc[:, ~X.columns.duplicated() & ~X.columns.str.startswith("__")]
tr, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
base = np.load("outputs/factory/confirm_combo10gnj.npz")["oof"]
m = lgb.LGBMClassifier(learning_rate=.1, n_estimators=300, num_leaves=31, colsample_bytree=.5, verbose=-1, n_jobs=30).fit(X.iloc[tr], y[tr])
order = pd.Series(m.booster_.feature_importance("gain"), X.columns).sort_values(ascending=False).index
import os
if os.environ.get("TABPFN_FEATS") == "basic":    # Kaggle S6E5 8th: TabPFN gets the plain features only (diversity)
    import pyarrow.parquet as pq
    keep = set(pq.read_schema("outputs/feat_train.parquet").names)
    order = order[order.isin(keep)]
Xa = X.replace([np.inf, -np.inf], np.nan)
if not os.environ.get("TABPFN_NAN"):             # docs: missing values are handled inside TabPFN, no imputation needed
    Xa = Xa.fillna(Xa.iloc[tr].median())
Xa = Xa.astype(np.float32)
P = []
for g in range(G):
    cols = order[g::G]
    clf = TabPFNClassifier(device="cuda", n_estimators=NEST, ignore_pretraining_limits=True, random_state=g,
                           inference_config={"SUBSAMPLE_SAMPLES": SUB} if SUB > 0 else None)
    clf.fit(Xa.iloc[tr][cols].to_numpy(), y[tr])
    P.append(clf.predict_proba(Xa.iloc[va][cols].to_numpy())[:, 1])
    print(f"group {g}: {len(cols)} cols  AUC {roc_auc_score(y[va], P[-1]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
yy, n = y[va], len(va)
rb = rankdata(base[va]) / n
rp = np.mean([rankdata(p) / n for p in P], 0)
rng = np.random.default_rng(0)
B = [rng.integers(0, n, n) for _ in range(300)]


def rep(nm, o):
    d = [roc_auc_score(yy[i], o[i]) - roc_auc_score(yy[i], rb[i]) for i in B]
    print(f"{nm:34s} AUC {roc_auc_score(yy, o):.4f}  gain {roc_auc_score(yy, o) - roc_auc_score(yy, rb):+.4f}  "
          f"CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")


print(f"TabPFN ensemble AUC {roc_auc_score(yy, rp):.4f}  LGB {roc_auc_score(yy, rb):.4f}  rank corr {np.corrcoef(rb, rp)[0, 1]:.3f}")
rep("equal rank blend", (rb + rp) / 2)
rep("LGB .8 + TabPFN .2", .8 * rb + .2 * rp)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
Z = np.column_stack([lg(base[va])] + [lg(p) for p in P])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
o0 = cross_val_predict(LogisticRegression(max_iter=500), Z[:, :1], yy, cv=cv, method="decision_function")
o1 = cross_val_predict(LogisticRegression(max_iter=500), Z, yy, cv=cv, method="decision_function")
rb = rankdata(o0) / n
rep("stack: LGB logit + TabPFN (logistic)", rankdata(o1) / n)
print(f"total {time.time() - t0:.0f}s")
