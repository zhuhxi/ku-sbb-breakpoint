"""2nd-place step 3-4 on one outer fold: X = XS + TabPFN column; selection LGB (750 trees) -> SHAP and gain rankings;
4 LGB (5000 trees, lr .01, colsample .2, bagging .8/4, depth 8; seeds 12/22/32/42) on SHAP top200 / gain top200 /
SHAP top500 / gain top500; mean probability.   python p2/model.py K  -> outputs/p2/oof_fold{K}.npz
"""
import sys, time
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

t0 = time.time(); K = int(sys.argv[1])
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy()
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
XS = pd.read_parquet("outputs/p2/xs.parquet")
Xtr, Xte = XS.loc[L.id].reset_index(drop=True), XS.loc[sub.id].reset_index(drop=True)
tr, va = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, y))[K]
T = np.load(f"outputs/p2/tabpfn_fold{K}.npz")
A, V, E = Xtr.iloc[tr].copy(), Xtr.iloc[va].copy(), Xte.copy()
A["col_0"], V["col_0"], E["col_0"] = T["tr"], T["va"], T["te"]
sel = lgb.LGBMClassifier(n_estimators=750, learning_rate=.01, colsample_bytree=.3, max_depth=8, random_state=42, verbosity=-1, n_jobs=30).fit(A, y[tr])
shap = np.abs(sel.predict_proba(A, pred_contrib=True)[:, :-1]).mean(0)
c1 = list(A.columns[np.argsort(shap)]); c2 = list(A.columns[np.argsort(sel.booster_.feature_importance("gain"))])
print(f"selection done {time.time() - t0:.0f}s", flush=True)
pv, pe = [], []
for seed, cols in ((12, c1[-200:]), (22, c2[-200:]), (32, c1[-500:]), (42, c2[-500:])):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=.01, colsample_bytree=.2, bagging_freq=4, bagging_fraction=.8, max_depth=8,
                           random_state=seed, verbosity=-1, n_jobs=30).fit(A[cols], y[tr])
    pv.append(m.predict_proba(V[cols])[:, 1]); pe.append(m.predict_proba(E[cols])[:, 1])
    print(f"  seed {seed} {len(cols)} cols: fold AUC {roc_auc_score(y[va], pv[-1]):.4f}  {time.time() - t0:.0f}s", flush=True)
p = np.mean(pv, 0)
print(f"fold {K}: TabPFN col alone {roc_auc_score(y[va], T['va']):.4f} | 2nd-place pipeline {roc_auc_score(y[va], p):.4f}")
np.savez(f"outputs/p2/oof_fold{K}.npz", va=va, p=p, te=np.mean(pe, 0))
