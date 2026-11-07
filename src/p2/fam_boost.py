"""1-minute family check: boost a small LightGBM from our OOF (init_score = logit of combo10gnj OOF) using ONLY one
family's novel 2nd-place columns; 5-fold, early stopping on an inner split. Gain = AUC(base + boost) - AUC(base), bootstrap CI.
A family that carries nothing beyond our model stops at ~0 trees and gains ~0."""
import glob, time
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
t0 = time.time()
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy()
g = np.clip(np.load("outputs/factory/confirm_combo10gnj.npz")["oof"], 1e-6, 1 - 1e-6); b0 = np.log(g / (1 - g))
rng = np.random.default_rng(0); BS = [rng.integers(0, len(y), len(y)) for _ in range(200)]
P = dict(objective="binary", learning_rate=.05, num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1,
         colsample_bytree=.5, n_estimators=500, verbose=-1, n_jobs=30)
for f in sorted(glob.glob("outputs/p2/novel_*.parquet")):
    X = pd.read_parquet(f).loc[L.id].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    o, nt = b0.copy(), []
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        a, b = train_test_split(tr, test_size=.15, stratify=y[tr], random_state=k)
        m = lgb.LGBMClassifier(**P).fit(X[a], y[a], init_score=b0[a], eval_set=[(X[b], y[b])], eval_init_score=[b0[b]],
                                        eval_metric="auc", callbacks=[lgb.early_stopping(30, verbose=False)])
        o[va] = b0[va] + m.predict(X[va], raw_score=True); nt.append(m.best_iteration_)
    d = [roc_auc_score(y[i], o[i]) - roc_auc_score(y[i], b0[i]) for i in BS]
    print(f"{f.split('novel_')[1][:-8]:10s} {X.shape[1]:4d} cols  gain {roc_auc_score(y, o) - roc_auc_score(y, b0):+.4f}  "
          f"CI [{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}]  trees {nt}  {time.time() - t0:.0f}s", flush=True)
