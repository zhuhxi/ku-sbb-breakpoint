"""RBF-SVM arm on each fold's top-K gain features (ranks, fold-clean selection as tabpfn_top.py); folds in parallel (joblib)."""
import sys, os, time, numpy as np, lightgbm as lgb
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
sys.path.insert(0, "probe"); import feats
K = int(sys.argv[1]); C = float(sys.argv[2]); G = sys.argv[3]; t0 = time.time()
X, Xt, y = feats.load(); Xn, XT = X.to_numpy(np.float32), Xt.to_numpy(np.float32); ALL = np.vstack([Xn, XT]); ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xn, y))
Q = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=2000, verbose=-1, n_jobs=8)
def run(k):
    tr, va = folds[k]; a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**Q, random_state=0).fit(Xn[a], y[a], eval_set=[(Xn[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    top = np.argsort(-m0.booster_.feature_importance("gain"))[:K]; R = np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in top]).astype(np.float32)
    s = SVC(C=C, gamma=G if G == "scale" else float(G), cache_size=4000).fit(R[tr], y[tr])
    return k, s.decision_function(R[va]), s.decision_function(R[len(y):])
oof = np.zeros(len(y)); te = np.zeros(len(XT))
for k, pv, pt in Parallel(n_jobs=5)(delayed(run)(k) for k in range(5)): oof[folds[k][1]] = pv; te += pt / 5
print(f"svm K{K} C{C} g{G} OOF {roc_auc_score(y, oof):.4f} {time.time()-t0:.0f}s"); np.save(f"outputs/arms/svm{K}_{C}_oof.npy", oof); np.save(f"outputs/arms/svm{K}_{C}_test.npy", te)
