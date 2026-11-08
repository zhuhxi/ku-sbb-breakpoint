"""Context-aware stacking over the arm groups of blendH (OOF ranks) + context (n, n0 share, segment lengths, family probs,
p(no break), noise type, rescaled flag). Second level is evaluated with an outer 5-fold CV on the OOF table (different seed),
so the comparison with the fixed-weight blend is fair. Models: logistic (ranks + ranks x context) and small LightGBM."""
import numpy as np, pandas as pd, glob, lightgbm as lgb
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy(); r = lambda a: rankdata(a) / len(a)
I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
G = {"pair": r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")))), "inter": r(I[2]),
     "revpair": r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_prev*.npz")))),
     "rctx": r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*rctx*_oof.npy")) if "noTTA" not in f and "mt" not in f)),
     "mt": r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*mt*_oof.npy")) if not any(x in f for x in ("rctx", "mtf", "mts", "mtb2")))),
     "rctxmt": r(np.load("outputs/arms/tpf60rctxmt_oof.npy"))}
H = r(2 * G["pair"] + G["inter"] + 2 * G["revpair"] + 6 * G["rctx"] + 9 * G["mt"] + 3 * G["rctxmt"])
A = pd.read_parquet("outputs/factory/semiauto_bag6w.parquet").reindex(L.id); F = pd.read_parquet("outputs/feat_train.parquet").reindex(L.id)
mb = pd.read_parquet("outputs/magic_b_train.parquet").reindex(L.id)
S = pd.read_parquet("data/train_series.parquet"); g = S.groupby("id"); n = g.size().reindex(L.id).to_numpy(); n0 = g["period"].apply(lambda v: (v == 0).sum()).reindex(L.id).to_numpy()
C = np.column_stack([np.log(n), n0 / n, np.log(np.minimum(n0, n - n0))] + [A[c].to_numpy() for c in A if c.startswith("saf_family_")] +
                    [A["saf_kind_None"].to_numpy(), A["saf_ntype_t"].to_numpy(), A["saf_garch"].to_numpy(), mb["mb_rescaled"].fillna(0).to_numpy()])
C = (C - C.mean(0)) / (C.std(0) + 1e-9)
Z = logit(np.clip(np.column_stack(list(G.values())), 1e-4, 1 - 1e-4)); Zh = logit(np.clip(H, 1e-4, 1 - 1e-4))[:, None]
X_lr = np.column_stack([Z, Zh, C, Zh * C]); X_gb = np.column_stack([Z, Zh, C])
cv = list(StratifiedKFold(5, shuffle=True, random_state=11).split(y, y)); o_lr = np.zeros(len(y)); o_gb = np.zeros(len(y))
for tr, va in cv:
    o_lr[va] = LogisticRegression(C=.1, max_iter=2000).fit(X_lr[tr], y[tr]).decision_function(X_lr[va])
    o_gb[va] = lgb.LGBMClassifier(n_estimators=300, learning_rate=.02, num_leaves=7, min_child_samples=200, subsample=.8, subsample_freq=1, colsample_bytree=.7, verbose=-1, n_jobs=16).fit(X_gb[tr], y[tr]).predict_proba(X_gb[va])[:, 1]
fa = lambda s: np.mean([roc_auc_score(y[va], s[va]) for _, va in cv]); rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(300)]
print(f"fixed blend H: {roc_auc_score(y, H):.4f} (mean fold {fa(H):.4f})")
for nm, s in (("logistic ctx", o_lr), ("lgb ctx", o_gb), ("avg(H, lgb ctx)", (r(H) + r(o_gb)) / 2)):
    d = [roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], H[i]) for i in B]
    print(f"{nm:16s} {roc_auc_score(y, s):.4f} (mean fold {fa(s):.4f})  vs H {roc_auc_score(y, s) - roc_auc_score(y, H):+.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]  folds up {sum(roc_auc_score(y[va], s[va]) > roc_auc_score(y[va], H[va]) for _, va in cv)}/5")
