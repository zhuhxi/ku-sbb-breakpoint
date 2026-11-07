"""Fast fold-0 check: top-200 picked from the combo10g columns vs top-200 picked from combo10g + every other factory
family on disk (old families that did not help when added on top of everything). lr 0.1, two seeds.

    python probe/pool_quick.py [K]
"""
import glob
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

K = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sys.argv = [sys.argv[0], "x"]
exec(open("probe/param_quick.py").read().split("P = dict(")[0].replace("name, vals = sys.argv[1], [float(v) for v in sys.argv[2:]]", ""))
files = [f for f in sorted(glob.glob("outputs/factory/*.parquet")) if not any(s in f for s in ("_test", "combo", "_train"))]
extra = pd.concat([ld(f).add_prefix(f.split("/")[-1][:-8] + "__") for f in files], axis=1)
extra = extra.loc[:, [c.split("__", 1)[1] not in X.columns for c in extra.columns]]
extra = extra.loc[:, extra.notna().mean() > 0.5]
pool = pd.concat([X, extra], axis=1)
print(f"combo10g {X.shape[1]} cols, pool {pool.shape[1]} cols ({len(files)} files), load {time.time() - t:.1f}s", flush=True)
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=30)


def fit(XX, a, b, s):
    m = lgb.LGBMClassifier(**P, random_state=s)
    m.fit(XX.iloc[a], y[a], eval_set=[(XX.iloc[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


for nm, XX in (("combo10g", X), ("pool", pool)):
    a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=0)
    imp = pd.Series(fit(XX, a, b, 0).booster_.feature_importance("gain"), index=XX.columns).sort_values(ascending=False)
    cols = list(imp.index[:K])
    aucs = [roc_auc_score(y[va], fit(XX[cols], a2, b2, s).predict_proba(XX.iloc[va][cols])[:, 1])
            for s, (a2, b2) in enumerate([train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=s) for s in range(2)])]
    new = sum(c not in X.columns for c in cols)
    print(f"top {K} from {nm:9s}: fold0 AUC {np.mean(aucs):.4f} (seeds {aucs[0]:.4f} {aucs[1]:.4f})  old-family cols picked {new}  {time.time() - t:.1f}s", flush=True)
    if new:
        print("   picked old families:", pd.Series([c.split("__")[0] for c in cols if c not in X.columns]).value_counts().head(8).to_dict())
