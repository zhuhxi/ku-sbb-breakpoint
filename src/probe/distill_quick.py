"""Distil the TabPFN average into LightGBM (CPU-cheap): regress logit(TabPFN OOF avg, rank->clipped) on the 919 features, 5-fold.
NOTE: targets for fold-train rows come from TabPFN models that saw the val fold (mild leak) -> optimistic upper bound."""
import sys, glob, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
X, Xt, y = feats.load(); X = X.to_numpy(np.float32); r = lambda a: rankdata(a) / len(a)
fs = [f for f in sorted(glob.glob("outputs/arms/tpf*_oof.npy")) if not any(x in f for x in ("tpf30_", "tpf150_", "ne16"))]
P = np.mean([np.clip(np.load(f), 1e-4, 1 - 1e-4) for f in fs], 0); tgt = logit(P)
mix = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0   # target = mix*logit(tabpfn) + (1-mix)*(+-3 label)
tg = mix * tgt + (1 - mix) * (6 * y - 3)
Q = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=4000, verbose=-1, n_jobs=28)
oof = np.zeros(len(y))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m = lgb.LGBMRegressor(**Q, random_state=0).fit(X[a], tg[a], eval_set=[(X[b], tg[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict(X[va])
print(f"mix {mix}: student OOF AUC {roc_auc_score(y, oof):.4f}  (TabPFN avg {roc_auc_score(y, P):.4f})  corr {np.corrcoef(r(oof), r(P))[0,1]:.3f}")
np.save(f"outputs/arms/distill{mix}_oof.npy", oof)
