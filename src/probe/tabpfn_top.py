"""TabPFN on each fold's top-K gain features (quick LGB on the fold's inner-train part), rank-transformed, whole fold-train as context.
    CUDA_VISIBLE_DEVICES=0 ~/venvs/tabpfn/bin/python probe/tabpfn_top.py K folds...   ;  ... merge K"""
import sys, time, os, numpy as np, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
MP = {"model_path": os.environ["MODEL"]} if os.environ.get("MODEL") else {}
if os.environ.get("TEMP"): MP["softmax_temperature"] = float(os.environ["TEMP"])
K = int(sys.argv[1]); y = feats.y; folds = list(StratifiedKFold(5, shuffle=True, random_state=int(os.environ.get("SPLIT", 42))).split(np.zeros(len(y)), y))
if os.environ.get("FULL"): folds = [(np.arange(len(y)), np.r_[np.where(y == 0)[0][:1], np.where(y == 1)[0][:1]])]   # test-only refit: context = ALL train rows; run with fold arg 0, TAG ...full
if sys.argv[2] == "merge":
    oof = np.zeros(len(y)); te = 0
    for k, (tr, va) in enumerate(folds): z = np.load(f"outputs/arms/tpf{K}{os.environ.get("TAG","")}_f{k}.npz"); oof[va] = z["va"]; te = te + z["test"] / 5
    print(f"TabPFN top{K}{os.environ.get("TAG","")} OOF {roc_auc_score(y, oof):.4f}"); np.save(f"outputs/arms/tpf{K}{os.environ.get("TAG","")}_oof.npy", oof); np.save(f"outputs/arms/tpf{K}{os.environ.get("TAG","")}_test.npy", te); sys.exit()
from tabpfn import TabPFNClassifier
if os.environ.get("REV"): feats.FAMS[-1] = "semiauto_bag6w"
if os.environ.get("SAFAM"): feats.FAMS[-1] = os.environ["SAFAM"]
if os.environ.get("EXTRA"): feats.FAMS += os.environ["EXTRA"].split(",")   # reversed features exist for bag6w only
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
    if os.environ.get("DECOR"):   # greedy in gain order, skip columns with |Spearman| > DECOR to an already chosen one (ranks on 4000 fold-train rows)
        from scipy.stats import rankdata as _rk; thr = float(os.environ["DECOR"]); sub_ = np.random.default_rng(k).choice(a, 4000, replace=False)
        order = np.argsort(-m0.booster_.feature_importance("gain")); Rk = {}; top = []
        for j in order:
            if len(top) == K: break
            rj = _rk(ALL[sub_, j]); rj = (rj - rj.mean()) / (rj.std() + 1e-9)
            if all(abs(np.mean(rj * Rk[t])) <= thr for t in top): top.append(j); Rk[j] = rj
        top = np.array(top)
    if os.environ.get("RSUB"):   # random subspace: K columns drawn from the top RSUB by gain (seeded by TS and fold)
        pool = np.argsort(-m0.booster_.feature_importance("gain"))[:int(os.environ["RSUB"])]; top = np.sort(np.random.default_rng(1000 * int(os.environ.get("TS", 0)) + k).choice(pool, K, replace=False))
    if os.environ.get("FORCE"):   # always include columns matching this regex
        fz = np.where(X.columns.str.contains(os.environ["FORCE"], regex=True))[0]; top = np.r_[top, [j for j in fz if j not in set(top)]].astype(int)
    R = np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in top]).astype(np.float32);
    if os.environ.get("PAIRSEL"):   # second stage: quick LGB on base + pair rank interactions of the top-60; TabPFN gets its top-K columns
        import itertools
        R60 = np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in np.argsort(-m0.booster_.feature_importance("gain"))[:60]]).astype(np.float32)
        PR = np.column_stack([R60[:, i] - R60[:, j] for i, j in itertools.combinations(range(60), 2)] + [R60[:, i] * R60[:, j] for i, j in itertools.combinations(range(60), 2)]).astype(np.float32)
        ALLR = np.column_stack([np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in range(ALL.shape[1])]).astype(np.float32), PR])
        m1 = lgb.LGBMClassifier(**Q, random_state=0).fit(ALLR[:n0][a], y[a], eval_set=[(ALLR[:n0][b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
        R = ALLR[:, np.argsort(-m1.booster_.feature_importance("gain"))[:K]]
    if os.environ.get("RAWV"):   # raw (median-filled) values instead of ranks
        R = ALL[:, top].astype(np.float32)
    SIDE = bool(os.environ.get("SIDE")) and bool(REV)
    SYM = bool(os.environ.get("SYM")) and bool(REV)
    if SYM:   # symmetric: original rows [own|reversed], reversed rows [reversed|own]; both in context; TTA over the two views
        Ro, Rv = R[:n0 + m0_], R[n0 + m0_:]; A1, A2 = np.hstack([Ro, Rv]), np.hstack([Rv, Ro])
        ctx2 = tr if not int(os.environ.get("CTX", 0)) else ctx
        clf = TabPFNClassifier(device="cuda", **MP, n_estimators=int(os.environ.get("NE", 8)), ignore_pretraining_limits=True, random_state=int(os.environ.get("TS", 0))).fit(np.vstack([A1[:n0][ctx2], A2[:n0][ctx2]]), np.r_[y[ctx2], y[ctx2]])
        pv = (clf.predict_proba(A1[:n0][va])[:, 1] + clf.predict_proba(A2[:n0][va])[:, 1]) / 2; pt = (clf.predict_proba(A1[n0:])[:, 1] + clf.predict_proba(A2[n0:])[:, 1]) / 2
        np.savez(f"outputs/arms/tpf{K}{os.environ.get('TAG','')}_f{k}.npz", va=pv, test=pt); print(f"fold {k}: {roc_auc_score(y[va], pv):.4f} {time.time()-t0:.0f}s", flush=True); continue
    if SIDE and os.environ.get("SIDECTX2"):   # extra context rows: the flipped view [rev|own] of the fold-train rows (same labels); predict only [own|rev]
        FL = np.hstack([R[n0 + m0_:n0 + m0_ + n0], R[:n0]])
    if SIDE:   # side-by-side: each original row gets its own reversed-series features as extra columns (no TTA, no reversed context)
        R = np.hstack([R[n0 + m0_:], R[:n0 + m0_]]) if os.environ.get("SIDEFLIP") else np.hstack([R[:n0 + m0_], R[n0 + m0_:]]) if not os.environ.get("SIDEDIFF") else (np.hstack([R[:n0 + m0_], R[:n0 + m0_] - R[n0 + m0_:]]) if os.environ["SIDEDIFF"] == "1" else np.hstack([R[:n0 + m0_], R[n0 + m0_:], R[:n0 + m0_] - R[n0 + m0_:]]))
    if os.environ.get("NORMSC"):   # normal scores instead of uniform ranks
        from scipy.stats import norm as _n; R = _n.ppf(np.clip(R, 1e-5, 1 - 1e-5)).astype(np.float32)
    Rtr, Rte = R[:n0], R[n0:n0 + m0_]
    CT = int(os.environ.get("CTX", 0)); ctx = tr if not CT else np.random.default_rng(k + 100 * int(os.environ.get("CTXSEED", 0))).choice(tr, CT, replace=False)
    if os.environ.get("FT"):   # fine-tune TabPFN on this fold's training part (internal 10% validation, time limit FT seconds)
        from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
        clf = FinetunedTabPFNClassifier(device="cuda", **MP, time_limit=int(os.environ["FT"]), epochs=int(os.environ.get("FTEP", 30)), learning_rate=float(os.environ.get("FTLR", 1e-5)),
                                        n_estimators_final_inference=int(os.environ.get("NE", 8)), n_finetune_ctx_plus_query_samples=int(os.environ.get("FTN", 10000)), n_estimators_finetune=1, n_estimators_validation=1, early_stopping_patience=int(os.environ.get("FTPAT", 8)), random_state=int(os.environ.get("TS", 0))).fit(Rtr[ctx], y[ctx])
    else:
        if os.environ.get("REG"):
            from tabpfn import TabPFNRegressor; clf = TabPFNRegressor(device="cuda", **MP, n_estimators=int(os.environ.get("NE", 8)), ignore_pretraining_limits=True, random_state=int(os.environ.get("TS", 0))).fit(Rtr[ctx], y[ctx].astype(float)); clf.predict_proba = lambda Z: np.column_stack([np.zeros(len(Z)), clf.predict(Z)])
        else:
            clf = TabPFNClassifier(device="cuda", **MP, n_estimators=int(os.environ.get("NE", 8)), ignore_pretraining_limits=True, random_state=int(os.environ.get("TS", 0))).fit(*((np.vstack([Rtr[ctx], FL[ctx]]), np.r_[y[ctx], y[ctx]]) if SIDE and os.environ.get("SIDECTX2") else (Rtr[ctx], y[ctx])))
    pv = clf.predict_proba(Rtr[va])[:, 1]; pt = clf.predict_proba(Rte)[:, 1]
    if os.environ.get("PSEUDO"):   # fold-clean transductive context: confident test rows (by this fold's model) join the context
        q = float(os.environ["PSEUDO"]); lo, hi = np.quantile(pt, q), np.quantile(pt, 1 - q); sel = (pt <= lo) | (pt >= hi)
        clf = TabPFNClassifier(device="cuda", **MP, n_estimators=int(os.environ.get("NE", 8)), ignore_pretraining_limits=True, random_state=int(os.environ.get("TS", 0))).fit(np.vstack([Rtr[ctx], Rte[sel]]), np.r_[y[ctx], (pt[sel] >= hi).astype(int)])
        pv = clf.predict_proba(Rtr[va])[:, 1]; pt = clf.predict_proba(Rte)[:, 1]
    if REV and not SIDE:
        Rr, Rtr2 = R[n0 + m0_:2 * n0 + m0_], R[2 * n0 + m0_:]
        if os.environ.get("CTXREV"):
            clf = TabPFNClassifier(device="cuda", **MP, n_estimators=int(os.environ.get("NE", 8)), ignore_pretraining_limits=True, random_state=int(os.environ.get("TS", 0))).fit(np.vstack([Rtr[ctx], Rr[ctx]]), np.r_[y[ctx], y[ctx]])
            pv = clf.predict_proba(Rtr[va])[:, 1]; pt = clf.predict_proba(Rte)[:, 1]
        np.savez(f"outputs/arms/tpf{K}{os.environ.get('TAG','')}noTTA_f{k}.npz", va=pv, test=pt)
        pv = (pv + clf.predict_proba(Rr[va])[:, 1]) / 2; pt = (pt + clf.predict_proba(Rtr2)[:, 1]) / 2
    np.savez(f"outputs/arms/tpf{K}{os.environ.get("TAG","")}_f{k}.npz", va=pv, test=pt); print(f"fold {k}: {roc_auc_score(y[va], pv):.4f} {time.time()-t0:.0f}s", flush=True)
