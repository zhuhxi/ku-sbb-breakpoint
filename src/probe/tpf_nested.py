"""Clean TabPFN teacher targets for distillation: for outer fold k, inner 4-fold CV on the outer-train part (top-K chosen on the
outer inner-train split as in tabpfn_top.py) -> teacher prob for every outer-train row, never using outer-val labels.
    CUDA_VISIBLE_DEVICES=0 ~/venvs/tabpfn/bin/python probe/tpf_nested.py K outer_folds... -> outputs/arms/tpfnest{K}_o{k}.npy"""
import sys, os, time, numpy as np, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
from tabpfn import TabPFNClassifier
K = int(sys.argv[1]); y = feats.y; t0 = time.time(); X, Xt, _ = feats.load(); Xn, XT = X.to_numpy(np.float32), Xt.to_numpy(np.float32)
ALL = np.vstack([Xn, XT]); ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL)
Q = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=2000, verbose=-1, n_jobs=16)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
for k in map(int, sys.argv[2:]):
    tr, va = folds[k]; a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**Q, random_state=0).fit(Xn[a], y[a], eval_set=[(Xn[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    top = np.argsort(-m0.booster_.feature_importance("gain"))[:K]; R = np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in top]).astype(np.float32)[:len(y)]
    T = np.full(len(y), np.nan)
    for itr, iva in StratifiedKFold(4, shuffle=True, random_state=k).split(tr, y[tr]):
        clf = TabPFNClassifier(device="cuda", n_estimators=8, ignore_pretraining_limits=True, random_state=0).fit(R[tr[itr]], y[tr[itr]])
        T[tr[iva]] = clf.predict_proba(R[tr[iva]])[:, 1]
    np.save(f"outputs/arms/tpfnest{K}_o{k}.npy", T); print(f"outer {k} done {time.time()-t0:.0f}s", flush=True)
