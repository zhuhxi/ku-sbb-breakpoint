"""Label-noise down-weighting (fold-clean): inside each outer fold, inner 3-fold quick OOF on the fold-train part; rows whose label
disagrees strongly (y=1 & p<LO or y=0 & p>1-LO) get weight W; retrain; compare with plain on the outer val fold."""
import sys, os, time, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
LO = float(sys.argv[1]); W = float(sys.argv[2]); t0 = time.time()
X, Xt, y = feats.load(); X = X.to_numpy(np.float32)
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=64)
fit = lambda a, b, w=None, s=0: lgb.LGBMClassifier(**P, random_state=s).fit(X[a], y[a], sample_weight=w, eval_set=[(X[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
o = {"plain": np.zeros(len(y)), "clean": np.zeros(len(y))}; fr = []
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    p = np.zeros(len(y))
    for itr, iva in StratifiedKFold(3, shuffle=True, random_state=k).split(tr, y[tr]):
        a, b = train_test_split(tr[itr], test_size=.1, stratify=y[tr[itr]], random_state=k); p[tr[iva]] = fit(a, b).predict_proba(X[tr[iva]])[:, 1]
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    bad = ((y == 1) & (p < LO)) | ((y == 0) & (p > 1 - LO)); w = np.where(bad, W, 1.0); fr.append(bad[tr].mean())
    o["plain"][va] = fit(a, b, None, 0).predict_proba(X[va])[:, 1]; o["clean"][va] = fit(a, b, w[a], 0).predict_proba(X[va])[:, 1]
rng = np.random.default_rng(0); d = [roc_auc_score(y[i], o["clean"][i]) - roc_auc_score(y[i], o["plain"][i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(200))]
print(f"LO {LO} W {W} flagged {np.mean(fr):.3f}: plain {roc_auc_score(y, o['plain']):.4f} clean {roc_auc_score(y, o['clean']):.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] {time.time()-t0:.0f}s")
