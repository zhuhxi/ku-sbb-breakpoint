"""Logistic arm on quantile-normal base features + per-fold top-K pair rank interactions (fold-clean), C from argv."""
import sys, os, time, itertools, numpy as np, lightgbm as lgb
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "probe"); import feats
K = int(sys.argv[1]); C = float(sys.argv[2]); t0 = time.time()
X, Xt, y = feats.load(); Xn, XT = X.to_numpy(np.float32), Xt.to_numpy(np.float32); ALL = np.vstack([Xn, XT]); ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL)
RA = np.column_stack([rankdata(ALL[:, j]) / (len(ALL) + 1) for j in range(ALL.shape[1])]).astype(np.float32); G = norm.ppf(RA).astype(np.float32)
Q = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=2000, verbose=-1, n_jobs=64)
pairs = list(itertools.combinations(range(K), 2)); oof = np.zeros(len(y)); te = np.zeros(len(XT))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xn, y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**Q, random_state=0).fit(Xn[a], y[a], eval_set=[(Xn[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    top = np.argsort(-m0.booster_.feature_importance("gain"))[:K]; R = RA[:, top] - .5
    Z = np.hstack([G] + ([np.column_stack([R[:, i] - R[:, j] for i, j in pairs] + [R[:, i] * R[:, j] * 4 for i, j in pairs])] if K else [])).astype(np.float32)
    lr = LogisticRegression(C=C, max_iter=300, solver="lbfgs").fit(Z[tr], y[tr]); oof[va] = lr.decision_function(Z[va]); te += lr.decision_function(Z[len(y):]) / 5
    print(f"fold {k} {roc_auc_score(y[va], oof[va]):.4f} {time.time()-t0:.0f}s", flush=True)
print(f"lrpair K{K} C{C} OOF {roc_auc_score(y, oof):.4f}"); np.save(f"outputs/arms/lrpair{K}_oof.npy", oof); np.save(f"outputs/arms/lrpair{K}_test.npy", te)
