"""Learning curve: quick LGB trained on fractions of the fold-train part, scored on the same val fold (folds 0,1)."""
import sys, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
X, Xt, y = feats.load(); X = X.to_numpy(np.float32)
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=32)
res = {}
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    if k > 1: break
    for f in (.5, .75, 1.0):
        for s in range(2):
            sub = tr if f == 1 else train_test_split(tr, train_size=f, stratify=y[tr], random_state=s)[0]
            a, b = train_test_split(sub, test_size=.1, stratify=y[sub], random_state=s)
            m = lgb.LGBMClassifier(**P, random_state=s).fit(X[a], y[a], eval_set=[(X[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
            res.setdefault(f, []).append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    print(k, {f: round(np.mean(v), 4) for f, v in res.items()}, flush=True)
n = {f: f * 19200 for f in res}; print({int(n[f]): round(np.mean(v), 4) for f, v in res.items()})
