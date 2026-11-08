"""Does time-reversal help? 242 features.py columns. Fold-clean: (a) base; (b) train on inner-train originals + their reversed copies
(same labels), early stop on originals; (c) TTA: mean of predictions on original and reversed val series."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]; y = lab.to_numpy()
F = pd.read_parquet("outputs/feat_train.parquet").reindex(lab.index); Rv = pd.read_parquet("outputs/rev_feat_train.parquet").reindex(lab.index)[F.columns]
X, XR = F.to_numpy(np.float32), Rv.to_numpy(np.float32)
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_estimators=4000, verbose=-1, n_jobs=30)
o = {k: np.zeros(len(y)) for k in ("base", "aug", "aug_tta", "base_tta")}
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**P, random_state=0).fit(X[a], y[a], eval_set=[(X[b], y[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    m1 = lgb.LGBMClassifier(**P, random_state=0).fit(np.vstack([X[a], XR[a]]), np.r_[y[a], y[a]], eval_set=[(X[b], y[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    o["base"][va] = m0.predict_proba(X[va])[:, 1]; o["base_tta"][va] = (o["base"][va] + m0.predict_proba(XR[va])[:, 1]) / 2
    o["aug"][va] = m1.predict_proba(X[va])[:, 1]; o["aug_tta"][va] = (o["aug"][va] + m1.predict_proba(XR[va])[:, 1]) / 2
    print(k, {kk: round(roc_auc_score(y[va], v[va]), 4) for kk, v in o.items()}, flush=True)
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(300)]
for kk in ("aug", "aug_tta", "base_tta"):
    d = [roc_auc_score(y[i], o[kk][i]) - roc_auc_score(y[i], o["base"][i]) for i in B]
    print(f"{kk}: {roc_auc_score(y, o[kk]):.4f} vs base {roc_auc_score(y, o['base']):.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]")
