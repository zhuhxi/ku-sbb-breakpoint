"""Error analysis of the CURRENT best blend (saf + bag6 + inter arms 1,2; public 0.85830), described with the semi-auto
estimates (what kind of series is it, what break would it be). 1) error rate / AUC by group; 2) calibration gaps: within
score deciles, which groups have more positives than the score says; 3) can any described variable predict the residual?
(logistic stack of logit(score) + one variable at a time, 5-fold, AUC gain; + all variables via LightGBM).
    python probe/err2.py"""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy(); ids = L.id.to_numpy()
r = lambda a: rankdata(a) / len(a)
O = lambda t: r(np.load(f"outputs/factory/confirm_{t}.npz")["oof"])
I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
s = (O("combo10gnjsaf") + O("combo10gnjbag6") + r(I[1]) + r(I[2])) / 4; s = r(s)
A = pd.read_parquet("outputs/factory/semiauto_bag6.parquet").loc[ids]
S = pd.read_parquet("data/train_series.parquet"); S["id"] = S["id"].astype(str)
g = S.groupby("id"); n = g.size().loc[ids].to_numpy(); n0 = g["period"].apply(lambda v: int((v == 0).sum())).loc[ids].to_numpy()
fam = A[[c for c in A if c.startswith("saf_family_")]].idxmax(1).str.replace("saf_family_", "").to_numpy()
kind = A[[c for c in A if c.startswith("saf_kind_") and not c.endswith("None")]].idxmax(1).str.replace("saf_kind_", "").to_numpy()
D = pd.DataFrame(dict(y=y, s=s, n=n, seg=np.minimum(n0, n - n0), frac=n0 / n, fam=fam, kind=kind,
                      p_break=1 - A.saf_kind_None.to_numpy(), size_lvl=A.saf_abs_lvl_d.to_numpy(), size_spr=A.saf_abs_log_k.to_numpy(),
                      size_dep=A.saf_abs_dep_d.to_numpy(), tnoise=A.saf_ntype_t.to_numpy(), garch=A.saf_garch.to_numpy(), decoy=A.saf_decoy.to_numpy(),
                      vburst=A.saf_vburst.to_numpy(), drift=A.saf_drift.to_numpy(), trend=A.saf_trend.to_numpy(), p1=A.saf_p1.to_numpy(), sinamp=A.saf_sin_amp.to_numpy()))
D["pred"] = (D.s > .5).astype(int); D["err"] = (D.pred != D.y).astype(int)
print(f"overall AUC {roc_auc_score(y, s):.4f}  error rate at median cut {D.err.mean():.3f}")
def grp(col, q=None):
    k = pd.qcut(D[col], q, duplicates="drop") if q else D[col]
    t = D.groupby(k, observed=True).apply(lambda d: pd.Series(dict(n=len(d), pos=d.y.mean(), auc=roc_auc_score(d.y, d.s) if d.y.nunique() > 1 else np.nan, err=d.err.mean(),
                                                                  miss=((d.y == 1) & (d.pred == 0)).sum() / max((d.y == 1).sum(), 1), false=((d.y == 0) & (d.pred == 1)).sum() / max((d.y == 0).sum(), 1))))
    print(f"\n-- by {col}"); print(t.round(3).to_string())
grp("fam"); grp("kind"); grp("seg", 5); grp("tnoise", 4); grp("garch", 4); grp("decoy", 4); grp("trend", 4)
print("\n-- residual check: does a variable add to logit(score)? (5-fold logistic, AUC gain)")
z = np.log(s / (1 - s + 1e-9) + 1e-9)[:, None]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
b = roc_auc_score(y, cross_val_predict(LogisticRegression(max_iter=500), z, y, cv=cv, method="decision_function"))
gains = {}
for c in ["n", "seg", "frac", "p_break", "size_lvl", "size_spr", "size_dep", "tnoise", "garch", "decoy", "vburst", "drift", "trend", "p1", "sinamp"]:
    v = r(D[c].to_numpy())[:, None] - .5; X = np.c_[z, v, z * v]
    gains[c] = roc_auc_score(y, cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=cv, method="decision_function")) - b
print({k: round(v, 4) for k, v in sorted(gains.items(), key=lambda kv: -kv[1])})
Xall = pd.get_dummies(D.drop(columns=["y", "pred", "err"]), columns=["fam", "kind"]).astype(float)
pa = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=.03, num_leaves=15, min_child_samples=100, verbose=-1, n_jobs=30), Xall, y, cv=cv, method="predict_proba")[:, 1]
print(f"LightGBM on score + all descriptors: {roc_auc_score(y, pa):.4f} vs score alone {roc_auc_score(y, s):.4f}")
