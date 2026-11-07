"""Covariate-shift weights: w = p(test|x)/p(train|x) (cross-fitted on train rows), clipped, mean 1. Quick LGB with vs without weights;
report plain OOF AUC and test-like (w-weighted) OOF AUC."""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "probe"); import feats
POW = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
t0 = time.time(); X, Xt, y = feats.load(); X, Xt = X.to_numpy(np.float32), Xt.to_numpy(np.float32)
A = np.vstack([X, Xt]); t = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
am = lgb.LGBMClassifier(n_estimators=80, learning_rate=.1, num_leaves=31, colsample_bytree=.5, min_child_samples=100, verbose=-1, n_jobs=32)
p = cross_val_predict(am, A, t, cv=StratifiedKFold(3, shuffle=True, random_state=0), method="predict_proba")[:len(X), 1]
w = (p / (1 - p)) * (len(X) / len(Xt)); w = np.clip(w, .2, 5) ** POW; w /= w.mean()
np.save("outputs/arms/adv_w.npy", w); print(f"weights q10/50/90 {np.percentile(w,[10,50,90]).round(2)}  {time.time()-t0:.0f}s", flush=True)
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=32)
oof = {"plain": np.zeros(len(y)), "weighted": np.zeros(len(y))}
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=k)
    for nm, sw in (("plain", None), ("weighted", w[a])):
        m = lgb.LGBMClassifier(**P, random_state=0).fit(X[a], y[a], sample_weight=sw, eval_set=[(X[b], y[b])], eval_sample_weight=[w[b]] if sw is not None else None, eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[nm][va] = m.predict_proba(X[va])[:, 1]
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(200)]
for lab, ww in (("plain AUC", None), ("test-like AUC", w)):
    f = lambda o, i: roc_auc_score(y[i], o[i], sample_weight=None if ww is None else ww[i])
    d = [f(oof["weighted"], i) - f(oof["plain"], i) for i in B]
    print(f"POW {POW} {lab}: plain {f(oof['plain'], slice(None)):.4f} weighted {f(oof['weighted'], slice(None)):.4f}  CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]")
print(f"{time.time()-t0:.0f}s")
