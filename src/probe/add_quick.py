"""Fast 5-fold check of adding one feature file to the combo10g columns (lr 0.1, one seed, same inner splits).

    python probe/add_quick.py outputs/factory/gnoise.parquet
    python probe/add_quick.py NEW.parquet --base-add outputs/factory/gnoise.parquet   # extra files already in the base
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

path = sys.argv[1]
base_add = sys.argv[sys.argv.index("--base-add") + 1:] if "--base-add" in sys.argv else []
sys.argv = [sys.argv[0], "x"]
exec(open("probe/param_quick.py").read().split("tr, va = next")[0].replace("name, vals = sys.argv[1], [float(v) for v in sys.argv[2:]]", ""))
N = ld(path)
if base_add:
    X = pd.concat([X] + [ld(f) for f in base_add], axis=1)
    print(f"base = combo10g + {base_add}: {X.shape[1]} cols")
print("alone AUC:", {c: round(max(roc_auc_score(y, N[c].fillna(N[c].median())), 1 - roc_auc_score(y, N[c].fillna(N[c].median()))), 4) for c in N.columns})
X2 = pd.concat([X, N], axis=1)
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=30)
oof = {"base": np.zeros(len(y)), "+new": np.zeros(len(y))}
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=k)
    for nm, XX in (("base", X), ("+new", X2)):
        m = lgb.LGBMClassifier(**P, random_state=0)
        m.fit(XX.iloc[a], y[a], eval_set=[(XX.iloc[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[nm][va] = m.predict_proba(XX.iloc[va])[:, 1]
    print(f"fold {k}: base {roc_auc_score(y[va], oof['base'][va]):.4f}  +new {roc_auc_score(y[va], oof['+new'][va]):.4f}  {time.time() - t:.0f}s", flush=True)
rng = np.random.default_rng(0)
d = [roc_auc_score(y[i], oof["+new"][i]) - roc_auc_score(y[i], oof["base"][i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(200))]
print(f"OOF base {roc_auc_score(y, oof['base']):.4f}  +new {roc_auc_score(y, oof['+new']):.4f}  gain {roc_auc_score(y, oof['+new']) - roc_auc_score(y, oof['base']):+.4f}  CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  {time.time() - t:.0f}s")
