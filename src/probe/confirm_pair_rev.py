"""Pair arm (as confirm_pair.py) + time reversal: ranks over original+reversed train/test rows; fold-train = originals + reversed
copies; val/test predictions = mean(original, reversed). REV root holds the reversed feature files (same layout).
    J=64 python probe/confirm_pair_rev.py K TAG FAM REV_ROOT"""
import sys, os, time, itertools, json, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
K, tag, fam, REV = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]; NS = 3; t0 = time.time(); y = feats.y
fams = feats.FAMS[:-1] + [fam]
X = feats.frame(feats.lab.index, "train", fams); X = X.loc[:, ~X.columns.duplicated()]; cols = X.columns
Xt = feats.frame(feats.sub["id"], "test", fams)[cols]
cwd = os.getcwd(); A = [X.to_numpy(np.float32), Xt.to_numpy(np.float32)]
for root in REV.split(","):   # one (train, test) pair of transformed copies per root
    os.chdir(root)
    A += [feats.frame(feats.lab.index, "train", fams).loc[:, lambda d: ~d.columns.duplicated()][cols].to_numpy(np.float32),
          feats.frame(feats.sub["id"], "test", fams).loc[:, lambda d: ~d.columns.duplicated()][cols].to_numpy(np.float32)]
    os.chdir(cwd)
n, m = len(y), len(Xt); NV = len(A) // 2
ALL = np.vstack(A); ALLf = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL); Xn = A[0]
J = int(os.environ.get("J", 64)); Q = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=J)
P = {**Q, "learning_rate": 0.02, "n_estimators": 5000}; pairs = list(itertools.combinations(range(K), 2))
print(f"{tag}: {X.shape[1]} base + {2*len(pairs)} pair cols, rows {ALL.shape[0]}", flush=True)
off = np.cumsum([0] + [len(a) for a in A]); sl = [slice(off[i], off[i + 1]) for i in range(len(A))]
oof = np.zeros(n); te = np.zeros(m)
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=int(os.environ.get("SPLIT", 42))).split(Xn, y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**Q, random_state=int(os.environ.get("TOPSEED", 0))).fit(Xn[a], y[a], eval_set=[(Xn[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    top = np.argsort(-m0.booster_.feature_importance("gain"))[:K]
    R = np.column_stack([(rankdata(ALLf[:, j]) / len(ALLf)).astype(np.float32) for j in top])
    Z = np.hstack([ALL, np.column_stack([R[:, i] - R[:, j] for i, j in pairs] + [R[:, i] * R[:, j] for i, j in pairs]).astype(np.float32)])
    Ztr_v = [Z[sl[2 * v]] for v in range(NV)]; Zte_v = [Z[sl[2 * v + 1]] for v in range(NV)]; Zo = Ztr_v[0]
    ms = []
    for s in range(NS):
        a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=100 * k + s)
        ms.append(lgb.LGBMClassifier(**P, random_state=100 * k + s).fit(np.vstack([zv[a] for zv in Ztr_v]), np.tile(y[a], NV), eval_set=[(Zo[b], y[b])], callbacks=[lgb.early_stopping(300, verbose=False)]))
    oof[va] = np.mean([np.mean([mm.predict_proba(zv[va])[:, 1] for zv in Ztr_v], 0) for mm in ms], 0)
    te += np.mean([np.mean([mm.predict_proba(zt)[:, 1] for zt in Zte_v], 0) for mm in ms], 0) / 5
    print(f"fold {k}: {roc_auc_score(y[va], oof[va]):.4f}  {time.time()-t0:.0f}s", flush=True)
print(f"{tag} OOF {roc_auc_score(y, oof):.4f}"); np.savez(f"outputs/factory/confirm_{tag}.npz", oof=oof, test=te)
