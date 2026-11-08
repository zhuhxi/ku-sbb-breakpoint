"""Plain LGB arm on the production set + extra factory files (fold-clean, 5-fold seed 42, inner 10% early stop, 2 seeds) -> outputs/arms/x{TAG}_{oof,test}.npy
    python probe/arm_lgb_extra.py TAG file1.parquet [file2 ...]"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
TAG, FILES = sys.argv[1], sys.argv[2:]; sys.path.insert(0, "probe"); import feats
X, XT, _ = feats.load(); y = feats.y
for f in FILES:
    D = pd.read_parquet(f); X = pd.concat([X, D.reindex(feats.lab.index)], axis=1); XT = pd.concat([XT, D.reindex(feats.sub["id"])], axis=1)
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=5000, verbose=-1, n_jobs=30)
o = np.zeros(len(y)); te = np.zeros(len(XT))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    for s in range(2):
        m = lgb.LGBMClassifier(**P, random_state=s).fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(200, verbose=False)])
        o[va] += m.predict_proba(X.iloc[va])[:, 1] / 2; te += m.predict_proba(XT)[:, 1] / 10
    print(f"fold {k}: {roc_auc_score(y[va], o[va]):.4f}", flush=True)
print(TAG, "OOF", round(roc_auc_score(y, o), 4)); np.save(f"outputs/arms/x{TAG}_oof.npy", o); np.save(f"outputs/arms/x{TAG}_test.npy", te)
