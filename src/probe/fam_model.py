"""Per-family specialist vs global quick LGB (same folds). Specialist trained only on its family's rows; final = specialist for that
family, global for the rest; also average of the two."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
X, Xt, y = feats.load(); A = pd.read_parquet("outputs/factory/semiauto_bag6.parquet").reindex(feats.lab.index)
fam = A[[c for c in A if c.startswith("saf_family_")]].idxmax(1).str.replace("saf_family_", "").to_numpy(); X = X.to_numpy(np.float32)
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=int(__import__("os").environ.get("J",32)))
def fit(idx, s=0):
    a, b = train_test_split(idx, test_size=.1, stratify=y[idx], random_state=s)
    return lgb.LGBMClassifier(**P, random_state=s).fit(X[a], y[a], eval_set=[(X[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
G = np.zeros(len(y)); S = np.zeros(len(y)); G2 = np.zeros(len(y)); T = sys.argv[1:] or ["WN"]
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    G[va] = fit(tr, k).predict_proba(X[va])[:, 1]; S[va] = G[va]; G2[va] = fit(tr, k + 50).predict_proba(X[va])[:, 1]
    for f in T:
        m = fit(tr[np.isin(fam[tr], [f])], k); v = va[fam[va] == f]; S[v] = m.predict_proba(X[v])[:, 1]
for f in T:
    v = fam == f; print(f, v.sum(), f"global {roc_auc_score(y[v], G[v]):.4f} specialist {roc_auc_score(y[v], S[v]):.4f} avg {roc_auc_score(y[v], (G[v]+S[v])/2):.4f}")
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(200)]
for nm, Z in (("specialist", S), ("avg", (G + S) / 2), ("CONTROL 2 global seeds", (G + G2) / 2)):
    d = [roc_auc_score(y[i], Z[i]) - roc_auc_score(y[i], G[i]) for i in B]
    if nm == "avg": d2 = [roc_auc_score(y[i], Z[i]) - roc_auc_score(y[i], (G[i] + G2[i]) / 2) for i in B]; print(f"  avg vs control: {roc_auc_score(y, Z) - roc_auc_score(y, (G + G2) / 2):+.4f} CI [{np.percentile(d2,2.5):+.4f},{np.percentile(d2,97.5):+.4f}]")
    print(f"overall {nm} {roc_auc_score(y, Z):.4f} vs global {roc_auc_score(y, G):.4f}  CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]")
