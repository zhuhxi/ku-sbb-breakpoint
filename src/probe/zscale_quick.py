"""Fast check: add length-scaled copies (x sqrt(n0 n1 / n)) of the raw two-sample distances (_dm _wass _ks _mw _lsr).
Fold 0, lr 0.1, two seeds, same as probe/param_quick.py.

    python probe/zscale_quick.py
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.argv = [sys.argv[0], "x"]
exec(open("probe/param_quick.py").read().split("P = dict(")[0].replace("name, vals = sys.argv[1], [float(v) for v in sys.argv[2:]]", ""))
S = pd.read_parquet("data/train_series.parquet", columns=["id", "period"])
n0 = S[S.period == 0].groupby("id").size().reindex(lab.index).to_numpy()
n1 = S[S.period == 1].groupby("id").size().reindex(lab.index).to_numpy()
h = np.sqrt(n0 * n1 / (n0 + n1)).astype(np.float32)
raw = [c for c in X.columns if c.endswith(("_dm", "_wass", "_ks", "_mw", "_lsr"))]
Z = X[raw].mul(h, axis=0).add_prefix("z_")
print(f"{len(raw)} raw distance columns scaled, {time.time() - t:.1f}s", flush=True)
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=30)
for nm, XX in (("base", X), ("base + scaled", pd.concat([X, Z], axis=1))):
    aucs = []
    for s in range(2):
        a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=s)
        m = lgb.LGBMClassifier(**P, random_state=s)
        m.fit(XX.iloc[a], y[a], eval_set=[(XX.iloc[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        aucs.append(roc_auc_score(y[va], m.predict_proba(XX.iloc[va])[:, 1]))
    sh = (n0[va] + n1[va]) < 500
    print(f"{nm:14s}: fold0 AUC {np.mean(aucs):.4f}  (seeds {aucs[0]:.4f} {aucs[1]:.4f})  {time.time() - t:.1f}s", flush=True)
