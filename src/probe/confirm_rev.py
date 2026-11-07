"""Production-style arm with time-reversal: LGB (lr .02, 3 seeds) trained on fold-train originals + reversed copies, predictions =
mean(original, reversed) for val and test (TTA). Feature families as rev_quick2.py. -> outputs/factory/confirm_{tag}.npz"""
import os, sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
ROOT, RREV, FAMS, tag = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]; NS = 3; t0 = time.time()
PAT = r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_|^saf_"
lab = pd.read_csv(f"{ROOT}/data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]; y = lab.to_numpy()
sub = pd.read_csv(f"{ROOT}/data/sample_submission.csv", dtype={"id": str})
def frame(root, idx, part):
    ld = lambda p: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    N = pd.concat([ld(f"{root}/outputs/factory/{f}.parquet") for f in FAMS], axis=1)
    import json; keep = json.load(open(f"{ROOT}/outputs/harvest_keep.json"))
    X = pd.concat([ld(f"{root}/outputs/feat_{part}.parquet"), ld(f"{root}/outputs/magic_b_{part}.parquet"), ld(f"{root}/outputs/harvest_{part}.parquet")[keep], N.loc[:, N.columns.str.contains(PAT, regex=True)]], axis=1); return X.loc[:, ~X.columns.duplicated()]
F = frame(ROOT, lab.index, "train"); cols = F.columns
X, XR = F.to_numpy(np.float32), frame(RREV, lab.index, "train")[cols].to_numpy(np.float32)
T, TR = frame(ROOT, sub["id"], "test")[cols].to_numpy(np.float32), frame(RREV, sub["id"], "test")[cols].to_numpy(np.float32)
P = dict(objective="binary", learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_estimators=6000, verbose=-1, n_jobs=int(os.environ.get("J", 64)))
AUG = int(os.environ.get("AUG", 1)); oof = np.zeros(len(y)); te = np.zeros(len(T))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    for s in range(NS):
        a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=100 * k + s)
        XA, yA = (np.vstack([X[a], XR[a]]), np.r_[y[a], y[a]]) if AUG else (X[a], y[a])
        m = lgb.LGBMClassifier(**P, random_state=100 * k + s).fit(XA, yA, eval_set=[(X[b], y[b])], callbacks=[lgb.early_stopping(300, verbose=False)])
        W = float(os.environ.get("TTA", 0.5)); oof[va] += ((1 - W) * m.predict_proba(X[va])[:, 1] + W * m.predict_proba(XR[va])[:, 1]) / NS; te += ((1 - W) * m.predict_proba(T)[:, 1] + W * m.predict_proba(TR)[:, 1]) / (5 * NS)
    print(f"fold {k}: {roc_auc_score(y[va], oof[va]):.4f} {time.time()-t0:.0f}s", flush=True)
print(f"{tag} OOF {roc_auc_score(y, oof):.4f}"); np.savez(f"{ROOT}/outputs/factory/confirm_{tag}.npz", oof=oof, test=te)
