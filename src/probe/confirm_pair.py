"""Production arm: production features (bag6w set) + per-fold pairwise rank interactions of the top-K gain features
(K chosen by a quick lr .1 model on the fold's inner-train part only; differences and products of ranks over train+test).
lr .02, 3 seeds, early stop 300 -> outputs/factory/confirm_{tag}.npz (oof, test in sample-submission order)."""
import sys, os, time, itertools, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
K = int(sys.argv[1]); tag = sys.argv[2]; fam = sys.argv[3] if len(sys.argv) > 3 else "semiauto_bag6w"; NS = 3; t0 = time.time()
X, Xt = feats.frame(feats.lab.index, "train", feats.FAMS[:-1] + [fam]), feats.frame(feats.sub["id"], "test", feats.FAMS[:-1] + [fam])
X = X.loc[:, ~X.columns.duplicated()]; Xt = Xt[X.columns]; y = feats.y
Xn, XT = X.to_numpy(np.float32), Xt.to_numpy(np.float32); ALL = np.vstack([Xn, XT]); ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL)
J = int(os.environ.get("J", 64)); Q = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=J)
P = {**Q, "learning_rate": 0.02, "n_estimators": 5000, "num_leaves": int(os.environ.get("NL", 31)), "min_child_samples": int(os.environ.get("MC", 40)), "reg_lambda": float(os.environ.get("RL", 1.0))}; pairs = list(itertools.combinations(range(K), 2))
print(f"{tag}: {X.shape[1]} base cols + {2*len(pairs)} pair cols", flush=True)
oof = np.zeros(len(y)); te = np.zeros(len(XT))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=int(os.environ.get("SPLIT", 42))).split(Xn, y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**Q, random_state=int(os.environ.get("TOPSEED", 0))).fit(Xn[a], y[a], eval_set=[(Xn[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    gi = m0.booster_.feature_importance("gain").astype(float)
    if os.environ.get("FSUB"):
        rs = np.random.default_rng(1000 * int(os.environ.get("TOPSEED", 0)) + k); gi[rs.random(len(gi)) > float(os.environ["FSUB"])] = -1
    if os.environ.get("EXCL"): gi[np.asarray(X.columns.str.contains(os.environ["EXCL"], regex=True))] = -1
    top = np.argsort(-gi)[:K]
    R = np.column_stack([(rankdata(ALL[:, j]) / len(ALL)).astype(np.float32) for j in top])
    Z = np.hstack(([R] if os.environ.get("NOBASE") else [np.vstack([Xn, XT])]) + [np.column_stack([R[:, i] - R[:, j] for i, j in pairs] + [R[:, i] * R[:, j] for i, j in pairs]).astype(np.float32)])
    Ztr, Zte = Z[:len(y)], Z[len(y):]
    ms = []
    for s in range(NS):
        a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=100 * k + s)
        ms.append(lgb.LGBMClassifier(**P, random_state=100 * k + s).fit(Ztr[a], y[a], eval_set=[(Ztr[b], y[b])], callbacks=[lgb.early_stopping(300, verbose=False)]))
    tp = np.mean([m.predict_proba(Zte)[:, 1] for m in ms], 0)
    if os.environ.get("PS"):   # fold-clean pseudo-labels: this fold's teachers label test; top/bottom 25% added; students replace teachers
        lo, hi = np.quantile(tp, .25), np.quantile(tp, .75); sel = (tp <= lo) | (tp >= hi); yp = (tp >= hi).astype(int)[sel]; ms2 = []
        for s in range(NS):
            a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=100 * k + s)
            ms2.append(lgb.LGBMClassifier(**P, random_state=100 * k + s).fit(np.vstack([Ztr[a], Zte[sel]]), np.r_[y[a], yp], eval_set=[(Ztr[b], y[b])], callbacks=[lgb.early_stopping(300, verbose=False)]))
        ms = ms2; tp = np.mean([m.predict_proba(Zte)[:, 1] for m in ms], 0)
    oof[va] = np.mean([m.predict_proba(Ztr[va])[:, 1] for m in ms], 0); te += tp / 5
    if os.environ.get("SAVE"):   # checkpoint: selected column indices + pair list + boosters, for CPU inference
        d = f"{os.environ['SAVE']}/{tag}/fold{k}"; os.makedirs(d, exist_ok=True); np.save(f"{d}/top.npy", top)
        for s, m in enumerate(ms): m.booster_.save_model(f"{d}/seed{s}.txt")
    print(f"fold {k}: {roc_auc_score(y[va], oof[va]):.4f}  {time.time()-t0:.0f}s", flush=True)
print(f"{tag} OOF {roc_auc_score(y, oof):.4f}"); np.savez(f"outputs/factory/confirm_{tag}.npz", oof=oof, test=te)
