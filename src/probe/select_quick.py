"""Fast 5-fold check of feature selection on the combo10g columns (lr 0.1, one seed). In each fold the full model is
fit on the training part; its gain importance (training part only) picks the top K columns, which are refit.

    python probe/select_quick.py 200 400
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

Ks = [int(k) for k in sys.argv[1:]]
sys.argv = [sys.argv[0], "x"]
exec(open("probe/param_quick.py").read().split("tr, va = next")[0].replace("name, vals = sys.argv[1], [float(v) for v in sys.argv[2:]]", ""))
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=30)
oof = {k: np.zeros(len(y)) for k in ["all"] + Ks}


def fit(cols, a, b, va):
    m = lgb.LGBMClassifier(**P, random_state=0)
    m.fit(X.iloc[a][cols], y[a], eval_set=[(X.iloc[b][cols], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, m.predict_proba(X.iloc[va][cols])[:, 1]


for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=k)
    m, oof["all"][va] = fit(list(X.columns), a, b, va)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    for K in Ks:
        _, oof[K][va] = fit(list(imp.index[:K]), a, b, va)
    print(f"fold {k}: " + "  ".join(f"{c}={roc_auc_score(y[va], o[va]):.4f}" for c, o in oof.items()) + f"  {time.time() - t:.0f}s", flush=True)
rng = np.random.default_rng(0)
B = [rng.integers(0, len(y), len(y)) for _ in range(200)]
for K in Ks:
    d = [roc_auc_score(y[i], oof[K][i]) - roc_auc_score(y[i], oof["all"][i]) for i in B]
    print(f"top {K}: OOF {roc_auc_score(y, oof[K]):.4f} vs all {roc_auc_score(y, oof['all']):.4f}  gain {roc_auc_score(y, oof[K]) - roc_auc_score(y, oof['all']):+.4f}  CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
print(f"{time.time() - t:.0f}s")
