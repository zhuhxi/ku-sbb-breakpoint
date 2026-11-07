"""Full brute-force interaction search, step 1: split, clean inputs, base-model residual.

    python brute/prep.py [--swap]

60/40 stratified split into a screening set and a holdout (--swap exchanges the roles).
Every transform below is fitted on screening rows only:
  raw  = value clipped to the screening 0.1/99.9% quantiles, NaN -> screening median
  rank = position in the sorted screening values, scaled to [-0.5, 0.5], NaN -> 0
The residual (label - out-of-fold base prediction) is what candidates are screened against.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import LGB_PARAMS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--swap", action="store_true")
args = ap.parse_args()
out = Path("outputs/brute") / ("swap" if args.swap else "main")
out.mkdir(parents=True, exist_ok=True)

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
load = lambda p, idx: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan)
F = pd.concat([load("outputs/feat_train.parquet", lab.index), load("outputs/raw_train.parquet", lab.index)], axis=1)
Ft = pd.concat([load("outputs/feat_test.parquet", sub["id"]), load("outputs/raw_test.parquet", sub["id"])], axis=1)[F.columns]
names = list(F.columns)
n_base = pd.read_parquet("outputs/feat_train.parquet").shape[1]   # first n_base columns = features.py

a, b = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=7)
screen, hold = (b, a) if args.swap else (a, b)

S = F.iloc[screen].to_numpy(np.float64)
lo, hi = np.nanquantile(S, 0.001, axis=0), np.nanquantile(S, 0.999, axis=0)
med = np.nanmedian(S, axis=0)
sorted_cols = [np.sort(c[~np.isnan(c)]) for c in S.T]


def transform(D):
    D = D.to_numpy(np.float64)
    raw = np.where(np.isnan(D), med, np.clip(D, lo, hi))
    rank = np.zeros_like(D)
    for j, s in enumerate(sorted_cols):
        r = np.searchsorted(s, D[:, j]) / max(len(s), 1) - 0.5
        rank[:, j] = np.where(np.isnan(D[:, j]), 0.0, r)
    return raw.astype(np.float32), rank.astype(np.float32)


raw, rank = transform(F)
raw_te, rank_te = transform(Ft)

# Out-of-fold base prediction on the screening set (features.py columns only = current pipeline).
Xs, ys = F.iloc[screen, :n_base], y[screen]
p = np.zeros(len(screen))
for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1).split(Xs, ys)):
    for s in range(3):
        m = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 600, "learning_rate": 0.03}, random_state=10 * f + s)
        p[va] += m.fit(Xs.iloc[tr], ys[tr]).predict_proba(Xs.iloc[va])[:, 1] / 3
resid = (ys - p).astype(np.float32)
print(f"screen {len(screen)}  holdout {len(hold)}  pool {len(names)} cols  base OOF corr(p, y) {np.corrcoef(p, ys)[0, 1]:.3f}")

np.savez(out / "prep.npz", raw=raw, rank=rank, raw_te=raw_te, rank_te=rank_te, y=y, screen=screen, hold=hold,
         resid=resid, names=np.array(names), n_base=n_base)
print("wrote", out / "prep.npz")
