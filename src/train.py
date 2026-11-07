"""Local KU-SBB pipeline: features -> stratified K-fold LightGBM -> OOF ROC-AUC -> submission.

    python train.py --data data            # real competition files
    python train.py --data data_fake       # smoke test (make_fake_data.py)

Early stopping uses an inner split of each training fold, so the reported OOF AUC
never sees the fold it scores.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.02, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
    n_estimators=5000, verbose=-1, n_jobs=16,
)


def cached_features(path, cache):
    if cache.exists() and cache.stat().st_mtime > max(path.stat().st_mtime, Path("features.py").stat().st_mtime):
        return pd.read_parquet(cache)
    t = time.time()
    feats = build_features(pd.read_parquet(path))
    feats.to_parquet(cache)
    print(f"features {path.name}: {feats.shape} in {time.time() - t:.0f}s")
    return feats


def fit_lgb(X, y, seed, params):
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
    m = lgb.LGBMClassifier(**params, random_state=seed)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc", callbacks=[lgb.early_stopping(300, verbose=False)])
    return m


def run_cv(Xtr, y, Xte, folds=5, seeds=(0,), params=LGB_PARAMS, logreg=True, verbose=True, select_k=None):
    oof = {"lgb": np.zeros(len(y)), "logreg": np.zeros(len(y))}
    pred = {"lgb": np.zeros(len(Xte)), "logreg": np.zeros(len(Xte))}
    importance = pd.Series(0.0, index=Xtr.columns)
    skf = StratifiedKFold(folds, shuffle=True, random_state=42)
    for k, (tr, va) in enumerate(skf.split(Xtr, y)):
        iters = []
        cols = list(Xtr.columns)
        if select_k:   # keep the top-k columns by gain of a fast model fit on this fold's training part only
            r = fit_lgb(Xtr.iloc[tr], y[tr], 100 * k, {**params, "learning_rate": 0.1})
            cols = list(pd.Series(r.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False).index[:select_k])
        for s in seeds:
            m = fit_lgb(Xtr.iloc[tr][cols], y[tr], 100 * k + s, params)
            oof["lgb"][va] += m.predict_proba(Xtr.iloc[va][cols])[:, 1] / len(seeds)
            pred["lgb"] += m.predict_proba(Xte[cols])[:, 1] / (folds * len(seeds))
            importance = importance.add(pd.Series(m.booster_.feature_importance("gain"), index=cols), fill_value=0)
            iters.append(m.best_iteration_)
        msg = f"fold {k}: lgb {roc_auc_score(y[va], oof['lgb'][va]):.4f} (iters {iters})"
        if logreg:
            # Preprocessing fit on the training fold only.
            lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
            Xf, Xt = Xtr.replace([np.inf, -np.inf], np.nan), Xte.replace([np.inf, -np.inf], np.nan)
            lr.fit(Xf.iloc[tr], y[tr])
            oof["logreg"][va] = lr.predict_proba(Xf.iloc[va])[:, 1]
            pred["logreg"] += lr.predict_proba(Xt)[:, 1] / folds
            msg += f"  logreg {roc_auc_score(y[va], oof['logreg'][va]):.4f}"
        if verbose:
            print(msg, flush=True)
    if not logreg:
        oof.pop("logreg"), pred.pop("logreg")
    return oof, pred, importance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1, help="LightGBM seeds averaged per fold")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    Xtr = cached_features(data / "train_series.parquet", out / "feat_train.parquet")
    Xte = cached_features(data / "test_series.parquet", out / "feat_test.parquet")
    labels = pd.read_csv(data / "train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    sub = pd.read_csv(data / "sample_submission.csv", dtype={"id": str})

    Xtr = Xtr.loc[labels.index]
    y = labels.to_numpy()
    Xte = Xte.reindex(sub["id"])[Xtr.columns]
    print(f"train {Xtr.shape}  test {Xte.shape}  pos rate {y.mean():.3f}")

    oof, pred, importance = run_cv(Xtr, y, Xte, args.folds, tuple(range(args.seeds)))
    metrics = {name: roc_auc_score(y, p) for name, p in oof.items()}
    print("OOF AUC:", {k: round(v, 4) for k, v in metrics.items()})

    sub["break_probability"] = np.clip(pred["lgb"], 0, 1)
    assert list(sub.columns) == ["id", "break_probability"] and np.isfinite(sub["break_probability"]).all()
    sub.to_csv(out / "submission.csv", index=False)
    pd.DataFrame({"id": labels.index, "label": y, **{f"oof_{k}": v for k, v in oof.items()}}).to_csv(out / "oof.csv", index=False)
    importance.sort_values(ascending=False).to_csv(out / "importance.csv", header=["gain"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out / 'submission.csv'}  top features: {list(importance.nlargest(10).index)}")


if __name__ == "__main__":
    main()
