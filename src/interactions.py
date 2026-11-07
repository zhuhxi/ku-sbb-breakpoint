"""Brute-force pairwise interaction search (datatech-style), done leak-free inside each CV fold.

    python interactions.py [--k 50] [--keep 100]

Per outer fold, using only that fold's training rows:
  1. rank base features by LightGBM gain, take the top K;
  2. build every pairwise a*b, a/b, b/a, a+b, a-b;
  3. keep the top `keep` interactions by gain in a model on base + interactions;
  4. train on base + kept interactions, score the untouched fold.
The base arm uses the same folds and seeds, so the two OOF predictions are paired.
"""
import argparse
from collections import Counter
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from train import LGB_PARAMS

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--keep", type=int, default=100)
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.to_numpy()
F = pd.read_parquet("outputs/feat_train.parquet").loc[lab.index]
S = pd.read_parquet("outputs/forensics_train.parquet").loc[lab.index]
glob = ["g_mean", "g_std", "g_cv", "g_absmax_over_std", "b_mean_raw", "b_std_raw", "n", "frac_b", "first_value", "last_value"]
X = pd.concat([F, S[glob].add_prefix("raw_")], axis=1).replace([np.inf, -np.inf], np.nan).astype(np.float32)
print(f"base pool {X.shape[1]} columns ({len(glob)} raw whole-series stats added)")

FAST = {**LGB_PARAMS, "n_estimators": 600, "learning_rate": 0.05, "colsample_bytree": 0.3}


def gain_rank(Xa, ya, seed):
    m = lgb.LGBMClassifier(**FAST, random_state=seed).fit(Xa, ya)
    return pd.Series(m.booster_.feature_importance("gain"), Xa.columns).sort_values(ascending=False)


def make_pairs(Xs, cols):
    out = {}
    for a, b in combinations(cols, 2):
        va, vb = Xs[a].to_numpy(), Xs[b].to_numpy()
        out[f"{a}*{b}"] = va * vb
        out[f"{a}/{b}"] = va / (np.abs(vb) + 1e-6) * np.sign(vb + 1e-12)
        out[f"{b}/{a}"] = vb / (np.abs(va) + 1e-6) * np.sign(va + 1e-12)
        out[f"{a}+{b}"] = va + vb
        out[f"{a}-{b}"] = va - vb
    return pd.DataFrame(out, index=Xs.index).astype(np.float32)


def fit_predict(Xtr_, ytr_, Xva_, seed):
    a, b = train_test_split(np.arange(len(ytr_)), test_size=0.1, stratify=ytr_, random_state=seed)
    m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    m.fit(Xtr_.iloc[a], ytr_[a], eval_set=[(Xtr_.iloc[b], ytr_[b])], eval_metric="auc",
          callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict_proba(Xva_)[:, 1]


oof_base, oof_int = np.zeros(len(y)), np.zeros(len(y))
picked = Counter()
for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    top = gain_rank(X.iloc[tr], y[tr], f).index[: args.k].tolist()
    P = make_pairs(X, top)                       # computed for all rows, selected on training rows only
    imp = gain_rank(pd.concat([X.iloc[tr], P.iloc[tr]], axis=1), y[tr], 100 + f)
    keep = [c for c in imp.index if c in P.columns][: args.keep]
    picked.update(keep)
    XI = pd.concat([X, P[keep]], axis=1)
    for s in range(args.seeds):
        oof_base[va] += fit_predict(X.iloc[tr], y[tr], X.iloc[va], 100 * f + s) / args.seeds
        oof_int[va] += fit_predict(XI.iloc[tr], y[tr], XI.iloc[va], 100 * f + s) / args.seeds
    print(f"fold {f}: {P.shape[1]} interactions from top {args.k}; base {roc_auc_score(y[va], oof_base[va]):.4f}  "
          f"+interactions {roc_auc_score(y[va], oof_int[va]):.4f}  best kept: {keep[:3]}", flush=True)

rng = np.random.default_rng(0)
d = [roc_auc_score(y[i], oof_int[i]) - roc_auc_score(y[i], oof_base[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(400))]
print(f"\nOOF base {roc_auc_score(y, oof_base):.4f}  +interactions {roc_auc_score(y, oof_int):.4f}  "
      f"diff 95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
rec = pd.Series(picked).sort_values(ascending=False)
rec.to_csv("outputs/interactions_picked.csv", header=["folds_kept"])
print("interactions kept in all 5 folds:", int((rec == 5).sum()))
print(rec.head(15).to_string())
