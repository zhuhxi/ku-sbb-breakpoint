"""Time-reversal on a bigger feature set: feat (features.py) + factory families (PAT-filtered), originals from ROOT, reversed from
ROOT_REV (same file layout). Fold-clean base / aug / TTA / aug+TTA, quick LGB (lr .05)."""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
ROOT, RREV = sys.argv[1], sys.argv[2]; FAMS = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []
PAT = r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_|^saf_"
lab = pd.read_csv(f"{ROOT}/data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]; y = lab.to_numpy()
def frame(root):
    ld = lambda p: pd.read_parquet(p).reindex(lab.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    import json; keep = json.load(open(f"{ROOT}/outputs/harvest_keep.json"))
    parts = [ld(f"{root}/outputs/feat_train.parquet"), ld(f"{root}/outputs/magic_b_train.parquet"), ld(f"{root}/outputs/harvest_train.parquet")[keep]]
    if FAMS:
        N = pd.concat([ld(f"{root}/outputs/factory/{f}.parquet") for f in FAMS], axis=1); parts.append(N.loc[:, N.columns.str.contains(PAT, regex=True)])
    X = pd.concat(parts, axis=1); return X.loc[:, ~X.columns.duplicated()]
F = frame(ROOT); R = frame(RREV)[F.columns]; X, XR = F.to_numpy(np.float32), R.to_numpy(np.float32); print("cols", X.shape[1], flush=True)
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_estimators=4000, verbose=-1, n_jobs=int(os.environ.get("J", 64)))
o = {k: np.zeros(len(y)) for k in ("base", "base_tta", "aug", "aug_tta")}
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**P, random_state=0).fit(X[a], y[a], eval_set=[(X[b], y[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    m1 = lgb.LGBMClassifier(**P, random_state=0).fit(np.vstack([X[a], XR[a]]), np.r_[y[a], y[a]], eval_set=[(X[b], y[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    o["base"][va] = m0.predict_proba(X[va])[:, 1]; o["base_tta"][va] = (o["base"][va] + m0.predict_proba(XR[va])[:, 1]) / 2
    o["aug"][va] = m1.predict_proba(X[va])[:, 1]; o["aug_tta"][va] = (o["aug"][va] + m1.predict_proba(XR[va])[:, 1]) / 2
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(300)]
for kk in ("base_tta", "aug", "aug_tta"):
    d = [roc_auc_score(y[i], o[kk][i]) - roc_auc_score(y[i], o["base"][i]) for i in B]
    print(f"{kk}: {roc_auc_score(y, o[kk]):.4f} vs base {roc_auc_score(y, o['base']):.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]", flush=True)
