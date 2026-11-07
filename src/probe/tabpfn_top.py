"""TabPFN on each fold's top-K gain features (quick LGB on the fold's inner-train part), rank-transformed, whole fold-train as context.
    CUDA_VISIBLE_DEVICES=0 ~/venvs/tabpfn/bin/python probe/tabpfn_top.py K folds...   ;  ... merge K"""
import sys, time, os, numpy as np, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
K = int(sys.argv[1]); y = feats.y; folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
if sys.argv[2] == "merge":
    oof = np.zeros(len(y)); te = 0
    for k, (tr, va) in enumerate(folds): z = np.load(f"outputs/arms/tpf{K}{os.environ.get("TAG","")}_f{k}.npz"); oof[va] = z["va"]; te = te + z["test"] / 5
    print(f"TabPFN top{K}{os.environ.get("TAG","")} OOF {roc_auc_score(y, oof):.4f}"); np.save(f"outputs/arms/tpf{K}{os.environ.get("TAG","")}_oof.npy", oof); np.save(f"outputs/arms/tpf{K}{os.environ.get("TAG","")}_test.npy", te); sys.exit()
from tabpfn import TabPFNClassifier
t0 = time.time(); X, Xt, _ = feats.load(); Xn, XT = X.to_numpy(np.float32), Xt.to_numpy(np.float32)
ALL = np.vstack([Xn, XT]); ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL)
Q = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=2000, verbose=-1, n_jobs=16)
for k in map(int, sys.argv[2:]):
    tr, va = folds[k]; a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**Q, random_state=int(os.environ.get("TS", 0))).fit(Xn[a], y[a], eval_set=[(Xn[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    top = np.argsort(-m0.booster_.feature_importance("gain"))[:K]
    R = np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in top]).astype(np.float32); Rtr, Rte = R[:len(y)], R[len(y):]
    CT = int(os.environ.get("CTX", 0)); ctx = tr if not CT else np.random.default_rng(k).choice(tr, CT, replace=False)
    clf = TabPFNClassifier(device="cuda", n_estimators=int(os.environ.get("NE", 8)), ignore_pretraining_limits=True, random_state=int(os.environ.get("TS", 0))).fit(Rtr[ctx], y[ctx])
    pv = clf.predict_proba(Rtr[va])[:, 1]; pt = clf.predict_proba(Rte)[:, 1]
    np.savez(f"outputs/arms/tpf{K}{os.environ.get("TAG","")}_f{k}.npz", va=pv, test=pt); print(f"fold {k}: {roc_auc_score(y[va], pv):.4f} {time.time()-t0:.0f}s", flush=True)
