"""Quick CV on the 4000-series subset: top-60 vs + structure-aware generator features (probe/gen_feats.py) vs
+ break-size features (probe/model_feats.py) vs + both, with paired bootstrap CIs against top-60.

    python probe/feats_combo_cv.py
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
A = pd.read_parquet("outputs/probe/gen_feats_4000.parquet")
B = pd.read_parquet("outputs/probe/model_feats_4000.parquet").reindex(A.index)
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
top = pd.read_csv("outputs/factory/perm_combo10.csv", index_col=0).index[:60].tolist()
srcs = ["outputs/feat_train.parquet", "outputs/magic_b_train.parquet", "outputs/harvest_train.parquet"] + \
    [f"outputs/factory/{f}.parquet" for f in ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")]
X = pd.concat([pd.read_parquet(s, columns=[c for c in pq.read_schema(s).names if c in top]) for s in srcs], axis=1)
X = X.loc[:, ~X.columns.duplicated()].reindex(A.index)
y = lab.reindex(A.index).to_numpy()
P = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=30, n_estimators=300,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=12)


def cv(Z, seeds=3):
    o = np.zeros(len(y))
    for s in range(seeds):
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(Z, y):
            o[va] += lgb.LGBMClassifier(**P, random_state=s).fit(Z.iloc[tr], y[tr]).predict_proba(Z.iloc[va])[:, 1] / seeds
    return o


o0 = cv(X)
boots = [np.random.default_rng(b).integers(0, len(y), len(y)) for b in range(300)]
print(f"top-60                    AUC {roc_auc_score(y, o0):.4f}")
for name, Z in [("+ structure-aware (old)", pd.concat([X, A], axis=1)), ("+ break-size (new)", pd.concat([X, B], axis=1)),
                ("+ both", pd.concat([X, A, B], axis=1))]:
    o = cv(Z)
    d = [roc_auc_score(y[i], o[i]) - roc_auc_score(y[i], o0[i]) for i in boots]
    print(f"{name:25s} AUC {roc_auc_score(y, o):.4f}  gain {roc_auc_score(y, o) - roc_auc_score(y, o0):+.4f}  95% CI {np.quantile(d, .025):+.4f} .. {np.quantile(d, .975):+.4f}")
