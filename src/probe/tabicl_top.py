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
    for k, (tr, va) in enumerate(folds): z = np.load(f"outputs/arms/ticl{K}{os.environ.get("TAG","")}_f{k}.npz"); oof[va] = z["va"]; te = te + z["test"] / 5
    print(f"TabICL top{K}{os.environ.get("TAG","")} OOF {roc_auc_score(y, oof):.4f}"); np.save(f"outputs/arms/ticl{K}{os.environ.get("TAG","")}_oof.npy", oof); np.save(f"outputs/arms/ticl{K}{os.environ.get("TAG","")}_test.npy", te); sys.exit()
from tabicl import TabICLClassifier
if os.environ.get("REV"): feats.FAMS[-1] = "semiauto_bag6w"   # reversed features exist for bag6w only
t0 = time.time(); X, Xt, _ = feats.load(); Xn, XT = X.to_numpy(np.float32), Xt.to_numpy(np.float32)
REV = os.environ.get("REV")   # reversed feature root: ranks over orig+rev, TTA, optional reversed context (CTXREV=1)
if REV:
    cwd = os.getcwd(); os.chdir(REV); XR, XTR, _ = feats.load(); os.chdir(cwd); XR, XTR = XR[X.columns].to_numpy(np.float32), XTR[X.columns].to_numpy(np.float32)
    ALL = np.vstack([Xn, XT, XR, XTR])
else: ALL = np.vstack([Xn, XT])
ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL); n0, m0_ = len(Xn), len(XT)
Q = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=2000, verbose=-1, n_jobs=16)
for k in map(int, sys.argv[2:]):
    tr, va = folds[k]; a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**Q, random_state=int(os.environ.get("TS", 0))).fit(Xn[a], y[a], eval_set=[(Xn[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    top = np.argsort(-m0.booster_.feature_importance("gain"))[:K]
    R = np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in top]).astype(np.float32);
    if os.environ.get("PAIRSEL"):   # second stage: quick LGB on base + pair rank interactions of the top-60; TabPFN gets its top-K columns
        import itertools
        R60 = np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in np.argsort(-m0.booster_.feature_importance("gain"))[:60]]).astype(np.float32)
        PR = np.column_stack([R60[:, i] - R60[:, j] for i, j in itertools.combinations(range(60), 2)] + [R60[:, i] * R60[:, j] for i, j in itertools.combinations(range(60), 2)]).astype(np.float32)
        ALLR = np.column_stack([np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in range(ALL.shape[1])]).astype(np.float32), PR])
        m1 = lgb.LGBMClassifier(**Q, random_state=0).fit(ALLR[:n0][a], y[a], eval_set=[(ALLR[:n0][b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
        R = ALLR[:, np.argsort(-m1.booster_.feature_importance("gain"))[:K]]
    Rtr, Rte = R[:n0], R[n0:n0 + m0_]
    CT = int(os.environ.get("CTX", 0)); ctx = tr if not CT else np.random.default_rng(k).choice(tr, CT, replace=False)
    clf = TabICLClassifier(device="cuda", n_estimators=int(os.environ.get("NE", 8)), random_state=int(os.environ.get("TS", 0))).fit(Rtr[ctx], y[ctx])
    pv = clf.predict_proba(Rtr[va])[:, 1]; pt = clf.predict_proba(Rte)[:, 1]
    if os.environ.get("PSEUDO"):   # fold-clean transductive context: confident test rows (by this fold's model) join the context
        q = float(os.environ["PSEUDO"]); lo, hi = np.quantile(pt, q), np.quantile(pt, 1 - q); sel = (pt <= lo) | (pt >= hi)
        clf = TabICLClassifier(device="cuda", n_estimators=int(os.environ.get("NE", 8)), random_state=int(os.environ.get("TS", 0))).fit(np.vstack([Rtr[ctx], Rte[sel]]), np.r_[y[ctx], (pt[sel] >= hi).astype(int)])
        pv = clf.predict_proba(Rtr[va])[:, 1]; pt = clf.predict_proba(Rte)[:, 1]
    if REV:
        Rr, Rtr2 = R[n0 + m0_:2 * n0 + m0_], R[2 * n0 + m0_:]
        if os.environ.get("CTXREV"):
            clf = TabICLClassifier(device="cuda", n_estimators=int(os.environ.get("NE", 8)), random_state=int(os.environ.get("TS", 0))).fit(np.vstack([Rtr[ctx], Rr[ctx]]), np.r_[y[ctx], y[ctx]])
            pv = clf.predict_proba(Rtr[va])[:, 1]; pt = clf.predict_proba(Rte)[:, 1]
        np.savez(f"outputs/arms/ticl{K}{os.environ.get('TAG','')}noTTA_f{k}.npz", va=pv, test=pt)
        pv = (pv + clf.predict_proba(Rr[va])[:, 1]) / 2; pt = (pt + clf.predict_proba(Rtr2)[:, 1]) / 2
    np.savez(f"outputs/arms/ticl{K}{os.environ.get("TAG","")}_f{k}.npz", va=pv, test=pt); print(f"fold {k}: {roc_auc_score(y[va], pv):.4f} {time.time()-t0:.0f}s", flush=True)
