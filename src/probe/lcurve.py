"""Learning curve: AUC on the same held-out folds when training on 25 / 50 / 100% of each training fold."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from train import fit_lgb, LGB_PARAMS
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]; y = lab.values
ld = lambda p: pd.read_parquet(p).reindex(lab.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
X = pd.concat([ld("outputs/feat_train.parquet"), ld("outputs/magic_b_train.parquet")], axis=1)
for frac in (0.25, 0.5, 1.0):
    oof = np.zeros(len(y))
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        rng = np.random.default_rng(k)
        sub = rng.choice(tr, int(len(tr) * frac), replace=False) if frac < 1 else tr
        for s in range(2):
            m = fit_lgb(X.iloc[sub], y[sub], 100 * k + s, LGB_PARAMS)
            oof[va] += m.predict_proba(X.iloc[va])[:, 1] / 2
    print(f"train size {int(len(y) * 0.8 * frac):6d}  AUC {roc_auc_score(y, oof):.4f}", flush=True)
