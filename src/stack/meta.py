"""Stacking level 1: learn how to combine the level-0 models, judged against the current blend.

    python stack/meta.py

Level-0 inputs (all OOF on StratifiedKFold(5, shuffle, 42); test = fold-averaged):
  ensemble.py (lgb, xgb, cat) · brute/lgb_inter.py --tag _B (base, all-91k, top-800) · cnn.py · stack/level0.py.
Every prediction becomes a rank in [0, 1]; context = series length, rescaled flag, tzc.
Meta-models are scored with their own CV (StratifiedKFold(5, shuffle, 7)) over the OOF table:
  current blend (rank mean of all-91k, top-800, GBDT ensemble) · equal mean of all · non-negative
  logistic · plain logistic · shallow LightGBM with context. Paired bootstrap vs the current blend,
  plain and on the test-like mix. Writes outputs/submission_stack.csv for the best stack.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

out = Path("outputs/stack")
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
r = lambda v: rankdata(v) / len(v)

oof, tst = {}, {}
E = pd.read_csv("outputs/oof_ens.csv", dtype={"id": str}).set_index("id").loc[lab.index]
Et = np.load("outputs/test_ens_models.npy")
for i, k in enumerate(("lgb", "xgb", "cat")):
    oof[k], tst[k] = E[f"oof_{k}"].to_numpy(), Et[i]
Io, It = np.load("outputs/brute/oof_lgb_inter_B.npy"), np.load("outputs/brute/test_lgb_inter_B.npy")
for i, k in enumerate(("inter_base", "all91k_B", "top800_B")):
    oof[k], tst[k] = Io[i], It[i]
oof["cnn"], tst["cnn"] = np.load("outputs/oof_cnn.npy"), np.load("outputs/pred_cnn.npy")
z = np.load(out / "level0_new.npz", allow_pickle=True)
for i, k in enumerate(z["names"]):
    oof[str(k)], tst[str(k)] = z["oof"][i], z["test"][i]
names = list(oof)
R = np.column_stack([r(oof[k]) for k in names])
Rt = np.column_stack([r(tst[k]) for k in names])

ctx_of = lambda feat, mb: np.column_stack([feat["n0"] + feat["n1"], mb["mb_rescaled"].fillna(0), mb["mb_tzc"].fillna(-1)])
C = ctx_of(pd.read_parquet("outputs/feat_train.parquet").loc[lab.index], pd.read_parquet("outputs/magic_b_train.parquet").reindex(lab.index))
Ct = ctx_of(pd.read_parquet("outputs/feat_test.parquet").loc[sub["id"]], pd.read_parquet("outputs/magic_b_test.parquet").reindex(sub["id"]))
flag = C[:, 1] == 1
pt, pr = Ct[:, 1].mean(), flag.mean()
w_mix = np.where(flag, pt / pr, (1 - pt) / (1 - pr))

print("level-0 models:")
for j, k in enumerate(names):
    print(f"  {k:18s} OOF AUC {roc_auc_score(y, oof[k]):.4f}")
cm = np.corrcoef(R.T)
print("rank correlation with all91k_B:", {k: round(cm[names.index('all91k_B'), j], 3) for j, k in enumerate(names)})


def nonneg_logistic(X, yy, l2=1e-3):
    Xc = X - 0.5
    def f(p):
        w, b = p[:-1], p[-1]
        s = Xc @ w + b
        loss = np.mean(np.logaddexp(0, s) - yy * s) + l2 * (w @ w)
        g = 1 / (1 + np.exp(-s)) - yy
        return loss, np.r_[Xc.T @ g / len(yy) + 2 * l2 * w, g.mean()]
    p = minimize(f, np.r_[np.full(X.shape[1], 0.1), 0.0], jac=True, method="L-BFGS-B",
                 bounds=[(0, None)] * X.shape[1] + [(None, None)]).x
    return p[:-1], p[-1]


def meta_lgb_fit(X, yy, seed):
    a, b = train_test_split(np.arange(len(yy)), test_size=0.15, stratify=yy, random_state=seed)
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=4, max_depth=2, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=seed)
    m.fit(X[a], yy[a], eval_set=[(X[b], yy[b])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


cur = [names.index(k) for k in ("all91k_B", "top800_B")]
gbdt = [names.index(k) for k in ("lgb", "xgb", "cat")]
current = lambda X: (X[:, cur].sum(1) + r(X[:, gbdt].mean(1))) / 3
arms = {"current blend": current(R), "equal mean of all": R.mean(1)}
o_nn, o_lr, o_lgb = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
XL = np.column_stack([R, C])
for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(R, y)):
    w, b = nonneg_logistic(R[tr], y[tr])
    o_nn[va] = (R[va] - 0.5) @ w + b
    o_lr[va] = LogisticRegression(C=1.0, max_iter=2000).fit(R[tr] - 0.5, y[tr]).decision_function(R[va] - 0.5)
    o_lgb[va] = meta_lgb_fit(XL[tr], y[tr], f).predict_proba(XL[va])[:, 1]
arms.update({"non-negative logistic": o_nn, "plain logistic": o_lr, "shallow LGB + context": o_lgb})

rng = np.random.default_rng(0)
idx = [rng.integers(0, len(y), len(y)) for _ in range(500)]
base = arms["current blend"]
res = {}
print("\nmeta-models (meta CV, paired bootstrap vs current blend):")
for k, p in arms.items():
    d = [roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], base[i]) for i in idx]
    dm = [roc_auc_score(y[i], p[i], sample_weight=w_mix[i]) - roc_auc_score(y[i], base[i], sample_weight=w_mix[i]) for i in idx]
    res[k] = (roc_auc_score(y, p), np.percentile(d, [2.5, 97.5]), roc_auc_score(y, p, sample_weight=w_mix), np.percentile(dm, [2.5, 97.5]))
    print(f"  {k:24s} AUC {res[k][0]:.4f}  CI [{res[k][1][0]:+.4f}, {res[k][1][1]:+.4f}]  | test-like mix {res[k][2]:.4f}  "
          f"CI [{res[k][3][0]:+.4f}, {res[k][3][1]:+.4f}]")

w, b = nonneg_logistic(R, y)
print("\nnon-negative logistic weights (full fit):", {k: round(v, 3) for k, v in zip(names, w) if v > 1e-4})

stacks = ("non-negative logistic", "plain logistic", "shallow LGB + context")
best = max(stacks, key=lambda k: res[k][0])
ok = res[best][1][0] > 0 and res[best][3][0] > 0
print(f"\nbest stack: {best}; beats current blend on plain AND test-like CI: {ok}")
if best == "non-negative logistic":
    pred = (Rt - 0.5) @ w + b
elif best == "plain logistic":
    pred = LogisticRegression(C=1.0, max_iter=2000).fit(R - 0.5, y).decision_function(Rt - 0.5)
else:
    XLt = np.column_stack([Rt, Ct])
    pred = np.mean([meta_lgb_fit(XL, y, s).predict_proba(XLt)[:, 1] for s in range(5)], 0)
s = sub.copy()
s["break_probability"] = r(pred)
assert s["id"].equals(sub["id"]) and np.isfinite(s["break_probability"]).all()
s.to_csv("outputs/submission_stack.csv", index=False)
print("wrote outputs/submission_stack.csv", "(adopt)" if ok else "(NOT recommended: did not pass both CIs)")
