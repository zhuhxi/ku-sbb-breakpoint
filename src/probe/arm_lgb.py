"""LGB arm on a feature parquet alone (or + production features with --plus), quick lr .05, 5-fold same split -> outputs/arms/{tag}_oof/test.npy
    python probe/arm_lgb.py tag file.parquet [--plus]"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
tag, path = sys.argv[1], sys.argv[2]; t0 = time.time(); y = feats.y
N = pd.read_parquet(path); X, Xt = N.reindex(feats.lab.index), N.reindex(feats.sub["id"])
if "--plus" in sys.argv:
    B, Bt, _ = feats.load(); X = pd.concat([B, X], axis=1); Xt = pd.concat([Bt, Xt], axis=1)
X, Xt = X.to_numpy(np.float32), Xt.to_numpy(np.float32)
P = dict(objective="binary", learning_rate=float(__import__("os").environ.get("LR", .05)), num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.3, reg_lambda=1.0, n_estimators=4000, verbose=-1, n_jobs=int(__import__("os").environ.get("J", 16)))
oof = np.zeros(len(y)); te = np.zeros(len(Xt))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=k)
    m = lgb.LGBMClassifier(**P, random_state=0).fit(X[a], y[a], eval_set=[(X[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict_proba(X[va])[:, 1]; te += m.predict_proba(Xt)[:, 1] / 5
    print(f"fold {k}: {roc_auc_score(y[va], oof[va]):.4f} {time.time()-t0:.0f}s", flush=True)
print(f"{tag} OOF {roc_auc_score(y, oof):.4f}"); np.save(f"outputs/arms/{tag}_oof.npy", oof); np.save(f"outputs/arms/{tag}_test.npy", te)
