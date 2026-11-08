"""Augmentation sets on 242 features.py columns, fold-clean: train on originals + chosen transformed copies, TTA over the same set."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]; y = lab.to_numpy()
F = pd.read_parquet("outputs/feat_train.parquet").reindex(lab.index); cols = F.columns
V = {"orig": F.to_numpy(np.float32), "rev": pd.read_parquet("outputs/rev_feat_train.parquet").reindex(lab.index)[cols].to_numpy(np.float32),
     "neg": pd.read_parquet("outputs/neg_feat_train.parquet").reindex(lab.index)[cols].to_numpy(np.float32),
     "revneg": pd.read_parquet("outputs/revneg_feat_train.parquet").reindex(lab.index)[cols].to_numpy(np.float32),
     "crop70": pd.read_parquet("outputs/crop70_feat_train.parquet").reindex(lab.index)[cols].to_numpy(np.float32),
     "inrev": pd.read_parquet("outputs/inrev_feat_train.parquet").reindex(lab.index)[cols].to_numpy(np.float32),
     "crop50": pd.read_parquet("outputs/crop50_feat_train.parquet").reindex(lab.index)[cols].to_numpy(np.float32)}
SETS = {"base": ["orig"], "rev": ["orig", "rev"], "inrev": ["orig", "inrev"], "rev+inrev": ["orig", "rev", "inrev"]}
TTA = {"base": ["orig"], "rev": ["orig", "rev"], "inrev": ["orig", "inrev"], "rev+inrev": ["orig", "rev", "inrev"]}
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_estimators=4000, verbose=-1, n_jobs=30)
o = {k: np.zeros(len(y)) for k in SETS}
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(V["orig"], y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    for nm, vs in SETS.items():
        m = lgb.LGBMClassifier(**P, random_state=0).fit(np.vstack([V[v][a] for v in vs]), np.tile(y[a], len(vs)), eval_set=[(V["orig"][b], y[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
        o[nm][va] = np.mean([m.predict_proba(V[v][va])[:, 1] for v in TTA[nm]], 0)
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(300)]
for nm in SETS:
    d = [roc_auc_score(y[i], o[nm][i]) - roc_auc_score(y[i], o["base"][i]) for i in B]
    print(f"{nm:6s} {roc_auc_score(y, o[nm]):.4f}  CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]")
