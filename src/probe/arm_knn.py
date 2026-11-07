"""kNN arm in the rank space of each fold's top-K gain features (fold-clean selection), distance-weighted, k from argv."""
import sys, os, time, numpy as np, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
sys.path.insert(0, "probe"); import feats
K = int(sys.argv[1]); NN = int(sys.argv[2]); t0 = time.time()
X, Xt, y = feats.load(); Xn, XT = X.to_numpy(np.float32), Xt.to_numpy(np.float32); ALL = np.vstack([Xn, XT]); ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL)
Q = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=2000, verbose=-1, n_jobs=64)
oof = np.zeros(len(y)); te = np.zeros(len(XT))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xn, y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**Q, random_state=0).fit(Xn[a], y[a], eval_set=[(Xn[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    g = m0.booster_.feature_importance("gain"); top = np.argsort(-g)[:K]; w = np.sqrt(g[top] / g[top].sum())
    R = np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in top]) * w
    kn = KNeighborsClassifier(NN, weights="distance", n_jobs=64).fit(R[tr], y[tr])
    oof[va] = kn.predict_proba(R[va])[:, 1]; te += kn.predict_proba(R[len(y):])[:, 1] / 5
print(f"knn K{K} n{NN} OOF {roc_auc_score(y, oof):.4f} {time.time()-t0:.0f}s"); np.save(f"outputs/arms/knn{K}_{NN}_oof.npy", oof); np.save(f"outputs/arms/knn{K}_{NN}_test.npy", te)
