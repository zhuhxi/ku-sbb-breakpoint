"""Break class only: which statistic tells real from fake (retuned three-bin)? Per-stat univariate AUC, and C2ST drop when the stat is removed."""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, "probe"); import gen_eval2 as G
t0 = time.time(); NF = 3000
R = G.real(); r = R[R.y == 1].drop(columns="y")
P = G.parse("outputs/factory/ku_gen_size_three-bin.json")
F = pd.DataFrame(Parallel(n_jobs=32, batch_size=32)(delayed(lambda s: G.stats(*G.gen(P, 900_000 + 2*s + 1, True)[:2]))(s) for s in range(NF)))
X = pd.concat([r, F]).replace([np.inf, -np.inf], np.nan); t = np.r_[np.zeros(len(r)), np.ones(len(F))]
c2 = lambda X: roc_auc_score(t, cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, verbose=-1, n_jobs=32), X, t, cv=5, method="predict_proba")[:, 1])
full = c2(X); print(f"C2ST all {len(X.columns)} stats: {full:.4f}")
rows = []
for c in X.columns:
    u = roc_auc_score(t, X[c].fillna(X[c].median())); rows.append((c, max(u, 1 - u), full - c2(X.drop(columns=c)),
                      "fake higher" if u > .5 else "fake lower", r[c].median(), F[c].median()))
D = pd.DataFrame(rows, columns=["stat", "uni_auc", "c2st_drop", "dir", "real_med", "fake_med"]).sort_values("c2st_drop", ascending=False)
print(D.round(4).to_string(index=False)); print(f"{time.time()-t0:.0f}s")
