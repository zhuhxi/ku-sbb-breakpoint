"""LightGBM + XGBoost + CatBoost on the cached features, same folds as train.py, equal-weight rank average.

    python train.py          # builds outputs/feat_*.parquet first
    python ensemble.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from train import LGB_PARAMS

out, data = Path("outputs"), Path("data")
labels = pd.read_csv(data / "train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv(data / "sample_submission.csv", dtype={"id": str})
Xtr = pd.read_parquet(out / "feat_train.parquet").loc[labels.index].replace([np.inf, -np.inf], np.nan)
Xte = pd.read_parquet(out / "feat_test.parquet").reindex(sub["id"])[Xtr.columns].replace([np.inf, -np.inf], np.nan)
y = labels.to_numpy()
SEEDS = (0, 1, 2)


def fit_predict(name, Xa, ya, Xb, yb, Xs, seed):
    if name == "lgb":
        m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc", callbacks=[lgb.early_stopping(300, verbose=False)])
    elif name == "xgb":
        m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.02, max_depth=5, min_child_weight=10,
                              subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0, eval_metric="auc",
                              early_stopping_rounds=300, tree_method="hist", n_jobs=16, random_state=seed)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    else:
        m = CatBoostClassifier(iterations=5000, learning_rate=0.03, depth=6, l2_leaf_reg=5, eval_metric="AUC",
                               thread_count=16, verbose=0, random_seed=seed, early_stopping_rounds=300)
        m.fit(Xa, ya, eval_set=(Xb, yb))
    return [m.predict_proba(X)[:, 1] for X in Xs]


models = ("lgb", "xgb", "cat")
oof = {k: np.zeros(len(y)) for k in models}
pred = {k: np.zeros(len(Xte)) for k in models}
for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, y)):
    for s in SEEDS:
        # Inner split for early stopping only; the scored fold is never seen.
        a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=100 * f + s)
        for k in models:
            pv, pt = fit_predict(k, Xtr.iloc[a], y[a], Xtr.iloc[b], y[b], [Xtr.iloc[va], Xte], 100 * f + s)
            oof[k][va] += pv / len(SEEDS)
            pred[k] += pt / (5 * len(SEEDS))
    print(f"fold {f}: " + "  ".join(f"{k} {roc_auc_score(y[va], oof[k][va]):.4f}" for k in models), flush=True)

np.save(out / "test_ens_models.npy", np.vstack([pred[k] for k in models]))  # per-model, fold-averaged (for stacking)
r = lambda v: rankdata(v) / len(v)
oof["ens"] = np.mean([r(oof[k]) for k in models], 0)
pred["ens"] = np.mean([r(pred[k]) for k in models], 0)
metrics = {k: roc_auc_score(y, v) for k, v in oof.items()}
print("OOF AUC:", {k: round(v, 4) for k, v in metrics.items()})

sub["break_probability"] = pred["ens"]
assert np.isfinite(sub["break_probability"]).all() and sub["break_probability"].between(0, 1).all()
sub.to_csv(out / "submission_ens.csv", index=False)
pd.DataFrame({"id": labels.index, "label": y, **{f"oof_{k}": v for k, v in oof.items()}}).to_csv(out / "oof_ens.csv", index=False)
(out / "metrics_ens.json").write_text(json.dumps(metrics, indent=2))
print("wrote", out / "submission_ens.csv")
