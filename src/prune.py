"""datatech-style permutation-importance pruning of the combo10 feature set.

    python prune.py --extra tt_RAW_DIFF_RES10_ARES10        # optionally add candidate families before pruning

Fast screen model (lr 0.05, same folds as screen.py, 1 seed). For each fold, every column of the validation
fold is permuted (2 repeats) and the AUC drop recorded; the drop is averaged over folds. Writes
outputs/factory/perm_{tag}.csv (mean drop, std over folds) and keep lists for thresholds 0 and +1e-5:
outputs/factory/keep_{tag}_{thr}.json. Screen a keep list with
    python screen.py <fams> --cols outputs/factory/keep_....json --tag ...
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
ap.add_argument("--extra", nargs="*", default=[])
ap.add_argument("--tag", default="combo10")
ap.add_argument("--repeats", type=int, default=2)
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
parts = [F, B, H[json.load(open("outputs/harvest_keep.json"))], N] + [ld(OUT / f"{f}.parquet") for f in args.extra]
X = pd.concat(parts, axis=1)
X = X.loc[:, ~X.columns.duplicated()]
print(f"{X.shape[1]} columns", flush=True)

t = time.time()
rng = np.random.default_rng(0)
drops = []
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=100 * k)
    m = lgb.LGBMClassifier(**P, random_state=100 * k)
    m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    Xv = X.iloc[va].to_numpy().copy()
    base = roc_auc_score(y[va], m.predict_proba(Xv)[:, 1])
    d = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = Xv[:, j].copy()
        for _ in range(args.repeats):
            Xv[:, j] = rng.permutation(col)
            d[j] += (base - roc_auc_score(y[va], m.predict_proba(Xv)[:, 1])) / args.repeats
        Xv[:, j] = col
    drops.append(d)
    print(f"fold {k}: AUC {base:.4f}  {time.time() - t:.0f}s", flush=True)

D = np.array(drops)
res = pd.DataFrame({"drop": D.mean(0), "sd": D.std(0), "pos_folds": (D > 0).sum(0)}, index=X.columns)
res.sort_values("drop", ascending=False).to_csv(OUT / f"perm_{args.tag}.csv")
for thr in (0.0, 1e-5):
    keep = res.index[res["drop"] > thr].tolist()
    json.dump(keep, open(OUT / f"keep_{args.tag}_{thr:g}.json", "w"))
    print(f"thr {thr:g}: keep {len(keep)} / {len(res)}")
print(res.sort_values("drop", ascending=False).head(20).round(5))
