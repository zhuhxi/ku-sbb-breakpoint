"""LGB variants as blend arms on the production features (quick, 5-fold same split): VARIANT in dart|xt|deep|shallow|goss|lin"""
import sys, os, time, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
v = sys.argv[1]; X, Xt, y = feats.load(); X, Xt = X.to_numpy(np.float32), Xt.to_numpy(np.float32); t0 = time.time()
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_estimators=3000, verbose=-1, n_jobs=int(os.environ.get("J", 64)))
P.update(dict(dart=dict(boosting_type="dart", n_estimators=600, learning_rate=.1, drop_rate=.1), xt=dict(extra_trees=True), deep=dict(num_leaves=255, min_child_samples=200, max_depth=-1),
              shallow=dict(num_leaves=4, n_estimators=8000, learning_rate=.1), goss=dict(data_sample_strategy="goss", subsample=1.0, subsample_freq=0), lin=dict(linear_tree=True, linear_lambda=1.0, num_leaves=15))[v])
oof = np.zeros(len(y)); te = np.zeros(len(Xt))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    cb = [] if v == "dart" else [lgb.early_stopping(100, verbose=False)]
    m = lgb.LGBMClassifier(**P, random_state=k).fit(X[a], y[a], eval_set=[(X[b], y[b])], eval_metric="auc", callbacks=cb)
    oof[va] = m.predict_proba(X[va])[:, 1]; te += m.predict_proba(Xt)[:, 1] / 5
print(f"{v} OOF {roc_auc_score(y, oof):.4f} {time.time()-t0:.0f}s"); np.save(f"outputs/arms/lgb_{v}_oof.npy", oof); np.save(f"outputs/arms/lgb_{v}_test.npy", te)
