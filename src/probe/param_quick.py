"""Fast direction check of one LightGBM parameter on the combo10g columns: fold 0 only, lr 0.1.

    python probe/param_quick.py colsample_bytree 0.5 0.2
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

t = time.time()
name, vals = sys.argv[1], [float(v) for v in sys.argv[2:]]
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.to_numpy()
ld = lambda p: pd.read_parquet(p).reindex(lab.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
N = pd.concat([ld(f"outputs/factory/{f}.parquet") for f in ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost", "genA", "genB")], axis=1)
N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_", regex=True)]
X = pd.concat([ld("outputs/feat_train.parquet"), ld("outputs/magic_b_train.parquet"),
               ld("outputs/harvest_train.parquet")[json.load(open("outputs/harvest_keep.json"))], N.loc[:, ~N.columns.duplicated()]], axis=1)
X = X.loc[:, ~X.columns.duplicated()]
tr, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
print(f"{X.shape[1]} cols, load {time.time() - t:.1f}s", flush=True)
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=30)
for v in vals:
    aucs = []
    for s in range(2):
        a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=s)
        m = lgb.LGBMClassifier(**{**P, name: type(P.get(name, v))(v)}, random_state=s)
        m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        aucs.append(roc_auc_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
    print(f"{name}={v:g}: fold0 AUC {np.mean(aucs):.4f}  (seeds {aucs[0]:.4f} {aucs[1]:.4f})  {time.time() - t:.1f}s", flush=True)
