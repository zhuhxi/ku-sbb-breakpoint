"""How much error is due to not knowing the series STRUCTURE? LightGBM on genR10 fakes (903 cols) vs the same + true nuisance
parameters (family, noise type/df, garch, sine amp/period, AR coefs, trend; never break parameters or label-linked nuisances).
Train bags a,b,e (90k), test bag f. Also per true kind / family."""
import os, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import train_test_split
E = pd.read_parquet("outputs/aug/extra_gnj_saf.parquet")
B = {b: pd.read_parquet(f"outputs/aug/X_simr10{b}.parquet") for b in "abef"}; M = {b: pd.read_parquet(f"outputs/aug/meta_simr10{b}.parquet").loc[B[b].index] for b in "abef"}
B = {b: pd.concat([v, E.reindex(v.index)], axis=1) for b, v in B.items()}
cols = [c for c in B["a"].columns if not c.startswith("mb_")]
NU = ["family", "ntype", "inv_df", "garch", "sin_amp", "sin_per", "p1", "p2", "trend"]
def nu(m):
    d = m[NU].copy(); d["family"] = d["family"].astype("category").cat.codes if d["family"].dtype == object else d["family"]; return d.astype(float).to_numpy(np.float32)
Xtr = np.r_[[B[b][cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32) for b in "abe"]].reshape(-1, len(cols)); ytr = np.r_[[M[b].label.to_numpy() for b in "abe"]].ravel()
Ntr = np.vstack([nu(M[b]) for b in "abe"]); Xte = B["f"][cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32); yte = M["f"].label.to_numpy(); Nte = nu(M["f"])
P = dict(objective="binary", learning_rate=.05, num_leaves=31, min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.5, reg_lambda=1., n_estimators=3000, verbose=-1, n_jobs=30)
def fit(X, t):
    a, b = train_test_split(np.arange(len(t)), test_size=.1, stratify=t, random_state=0)
    return lgb.LGBMClassifier(**P).fit(X[a], t[a], eval_set=[(X[b], t[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
pa = fit(Xtr, ytr).predict_proba(Xte)[:, 1]; pb = pa if os.environ.get("ABL") else fit(np.c_[Xtr, Ntr], ytr).predict_proba(np.c_[Xte, Nte])[:, 1]
print(f"A features only {A(yte, pa):.4f}   B + true structure {A(yte, pb):.4f}")
mf = M["f"]; neg = yte == 0
for k in ("spread", "level", "dep", "shape"):
    s = (mf.kind == k).to_numpy() | neg; print(f"  kind {k:7s} A {A(yte[s], pa[s]):.4f}  B {A(yte[s], pb[s]):.4f}")
for f in mf.family.unique():
    s = (mf.family == f).to_numpy(); print(f"  family {str(f):4s} n {s.sum():5d} A {A(yte[s], pa[s]):.4f}  B {A(yte[s], pb[s]):.4f}")
import os
if os.environ.get("ABL"):
    for nm, idx in (("noise type+df", [1, 2]), ("garch", [3]), ("family", [0]), ("AR coefs", [6, 7]), ("sine", [4, 5]), ("trend", [8])):
        p = fit(np.c_[Xtr, Ntr[:, idx]], ytr).predict_proba(np.c_[Xte, Nte[:, idx]])[:, 1]
        print(f"  + {nm:14s} {A(yte, p):.4f}  (+{A(yte, p) - A(yte, pa):.4f})", flush=True)
