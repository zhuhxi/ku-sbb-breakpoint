"""2nd-place pipeline with the selection step swapped: permutation importance on a held-out inner split (20% of the
outer-train part; AUC drop, 1 shuffle per column) instead of SHAP / gain. Then 4 LGB as in the original:
perm top200 (seeds 12, 22), perm top500 (seeds 32, 42), mean probability.   python p2/model_perm.py K
"""
import sys, time
import lightgbm as lgb, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

t0 = time.time(); K = int(sys.argv[1])
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy()
XS = pd.read_parquet("outputs/p2/xs.parquet").loc[L.id].reset_index(drop=True)
tr, va = list(StratifiedKFold(5, shuffle=True, random_state=42).split(XS, y))[K]
T = np.load(f"outputs/p2/tabpfn_fold{K}.npz")
A, V = XS.iloc[tr].copy(), XS.iloc[va].copy(); A["col_0"], V["col_0"] = T["tr"], T["va"]
ia, ib = train_test_split(np.arange(len(tr)), test_size=.2, stratify=y[tr], random_state=0)
sel = lgb.LGBMClassifier(n_estimators=750, learning_rate=.01, colsample_bytree=.3, max_depth=8, random_state=42, verbosity=-1, n_jobs=30).fit(A.iloc[ia], y[tr][ia])
B, yb = A.iloc[ib].to_numpy(), y[tr][ib]
base = roc_auc_score(yb, sel.predict_proba(B)[:, 1])
used = np.flatnonzero(sel.booster_.feature_importance("gain") > 0)
print(f"selection model: inner holdout AUC {base:.4f}, {len(used)} of {A.shape[1]} cols used  {time.time() - t0:.0f}s", flush=True)
bst = sel.booster_
def drop(j):
    Z = B.copy(); Z[:, j] = np.random.default_rng(j).permutation(Z[:, j])
    return base - roc_auc_score(yb, bst.predict(Z, num_threads=1))
imp = np.full(A.shape[1], -1.0); imp[used] = Parallel(n_jobs=30)(delayed(drop)(j) for j in used)
order = list(A.columns[np.argsort(imp)])
print(f"permutation done {time.time() - t0:.0f}s; cols with drop > 0: {(imp > 0).sum()}", flush=True)
pv = []
for seed, cols in ((12, order[-200:]), (22, order[-200:]), (32, order[-500:]), (42, order[-500:])):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=.01, colsample_bytree=.2, bagging_freq=4, bagging_fraction=.8, max_depth=8,
                           random_state=seed, verbosity=-1, n_jobs=30).fit(A[cols], y[tr])
    pv.append(m.predict_proba(V[cols])[:, 1])
    print(f"  seed {seed} {len(cols)} cols: fold AUC {roc_auc_score(y[va], pv[-1]):.4f}  {time.time() - t0:.0f}s", flush=True)
print(f"fold {K}: permutation-selected pipeline {roc_auc_score(y[va], np.mean(pv, 0)):.4f}  (SHAP/gain version .7891)")
np.savez(f"outputs/p2/oof_perm_fold{K}.npz", va=va, p=np.mean(pv, 0))
