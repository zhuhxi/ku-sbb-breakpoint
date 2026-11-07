"""Honest check of permutation pruning: the keep list is chosen inside each outer fold from its training part only.

    python prune_nested.py --thr 1e-5

Outer folds = StratifiedKFold(5, shuffle, 42) as everywhere. Inside the outer training part: 3 inner folds of the
fast screen model, permutation drop per column (1 repeat), averaged; keep drop > thr. Then the outer fold is
scored with the pruned set (2 seeds, same recipe as screen.py) and compared with the unpruned combo10 base OOF
(outputs/factory/base_oof_s2_combo10.npy). Writes outputs/factory/oof_c10_nested_{thr}.npy.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--thr", type=float, default=1e-5)
args = ap.parse_args()

P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, n_estimators=3000, verbose=-1, n_jobs=30)
OUT = Path("outputs/factory")
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.to_numpy()
ld = lambda p: pd.read_parquet(p).reindex(lab.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
F, B, H = ld("outputs/feat_train.parquet"), ld("outputs/magic_b_train.parquet"), ld("outputs/harvest_train.parquet")
N = pd.concat([ld(OUT / f"{f}.parquet") for f in ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")], axis=1)
N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$", regex=True)]
X = pd.concat([F, B, H[json.load(open("outputs/harvest_keep.json"))], N], axis=1)
X = X.loc[:, ~X.columns.duplicated()]
Xn = X.to_numpy()


def fit(idx, cols, seed):
    a, b = train_test_split(idx, test_size=0.1, stratify=y[idx], random_state=seed)
    m = lgb.LGBMClassifier(**P, random_state=seed)
    m.fit(Xn[np.ix_(a, cols)], y[a], eval_set=[(Xn[np.ix_(b, cols)], y[b])], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


t = time.time()
rng = np.random.default_rng(0)
allc = np.arange(X.shape[1])
oof = np.zeros(len(y))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    drop = np.zeros(X.shape[1])
    for j, (itr, iva) in enumerate(StratifiedKFold(3, shuffle=True, random_state=k).split(tr, y[tr])):
        itr, iva = tr[itr], tr[iva]
        m = fit(itr, allc, 1000 + 10 * k + j)
        Xv = Xn[iva].copy()
        base = roc_auc_score(y[iva], m.predict_proba(Xv)[:, 1])
        for c in allc:
            col = Xv[:, c].copy()
            Xv[:, c] = rng.permutation(col)
            drop[c] += (base - roc_auc_score(y[iva], m.predict_proba(Xv)[:, 1])) / 3
            Xv[:, c] = col
    keep = allc[drop > args.thr]
    for s in range(2):
        m = fit(tr, keep, 100 * k + s)
        oof[va] += m.predict_proba(Xn[np.ix_(va, keep)])[:, 1] / 2
    print(f"outer {k}: keep {len(keep)}  fold AUC {roc_auc_score(y[va], oof[va]):.4f}  {time.time() - t:.0f}s", flush=True)

o0 = np.load(OUT / "base_oof_s2_combo10.npy")
d = []
for _ in range(300):
    i = rng.integers(0, len(y), len(y))
    d.append(roc_auc_score(y[i], oof[i]) - roc_auc_score(y[i], o0[i]))
a0, a1 = roc_auc_score(y, o0), roc_auc_score(y, oof)
print(f"nested prune thr {args.thr:g}: OOF {a0:.4f} -> {a1:.4f} ({a1 - a0:+.4f}, CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}])")
np.save(OUT / f"oof_c10_nested_{args.thr:g}.npy", oof)
