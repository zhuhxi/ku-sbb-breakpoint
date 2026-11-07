"""Clean distillation: outer fold k student LGB regresses mix*logit(teacher_k) + (1-mix)*(6y-3) on outer-train rows, where teacher_k
comes from tpf_nested.py (never saw outer-val labels). OOF AUC + blend with the pair blend."""
import sys, glob, os, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
K = int(sys.argv[1]); mix = float(sys.argv[2]); X, Xt, y = feats.load(); X = X.to_numpy(np.float32); r = lambda a: rankdata(a) / len(a)
Q = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=4000, verbose=-1, n_jobs=int(os.environ.get("J", 28)))
oof = np.zeros(len(y))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    T = np.load(f"outputs/arms/tpfnest{K}_o{k}.npy"); tg = mix * logit(np.clip(T, 1e-4, 1 - 1e-4)) + (1 - mix) * (6 * y - 3)
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m = lgb.LGBMRegressor(**Q, random_state=0).fit(X[a], tg[a], eval_set=[(X[b], tg[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict(X[va]); print(f"fold {k} {roc_auc_score(y[va], oof[va]):.4f}", flush=True)
I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy"); pa = r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz"))))
base = r(2 * pa + r(I[2])); ta = r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*_oof.npy")) if not any(x in f for x in ("tpf30_", "tpf150_", "ne16"))))
print(f"K{K} mix {mix}: clean student OOF {roc_auc_score(y, oof):.4f} | pair blend {roc_auc_score(y, base):.4f} -> +student x2 {roc_auc_score(y, r(2*pa + r(I[2]) + 2*r(oof))):.4f} x4 {roc_auc_score(y, r(2*pa + r(I[2]) + 4*r(oof))):.4f} | with TabPFN avg x6 {roc_auc_score(y, r(2*pa + r(I[2]) + 6*ta)):.4f}")
np.save(f"outputs/arms/dist{K}_{mix}_oof.npy", oof)
