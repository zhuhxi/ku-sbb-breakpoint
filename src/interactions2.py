"""Larger leak-free interaction search on features.py + the raw-scale pool (rawstats.py).

    python interactions2.py [--k 80] [--keep 150]

Per outer fold, using only that fold's training rows: rank the pool by gain, take the top K,
build a*b, a/b, b/a, a+b, a-b, a*b^2, a^2*b for every pair, keep the top `keep` interactions,
then score the untouched fold. Arms share folds and seeds: F only, F+raw, F+raw+interactions.
Writes the per-fold kept lists so stable interactions can be reused.
"""
import argparse
import json
from collections import Counter
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from train import LGB_PARAMS

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=80)
ap.add_argument("--keep", type=int, default=150)
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.to_numpy()
F = pd.read_parquet("outputs/feat_train.parquet").loc[lab.index]
R = pd.read_parquet("outputs/raw_train.parquet").loc[lab.index]
clean = lambda D: D.replace([np.inf, -np.inf], np.nan).astype(np.float32)
F, R = clean(F), clean(R)
POOL = pd.concat([F, R], axis=1)
RANK = {**LGB_PARAMS, "n_estimators": 500, "learning_rate": 0.05, "colsample_bytree": 0.15}


def gain_rank(D, yy, seed):
    m = lgb.LGBMClassifier(**RANK, random_state=seed).fit(D, yy)
    return pd.Series(m.booster_.feature_importance("gain"), D.columns).sort_values(ascending=False)


def make_pairs(D, cols):
    out = {}
    for a, b in combinations(cols, 2):
        va, vb = D[a].to_numpy(np.float64), D[b].to_numpy(np.float64)
        out[f"{a}*{b}"], out[f"{a}+{b}"], out[f"{a}-{b}"] = va * vb, va + vb, va - vb
        out[f"{a}/{b}"] = va / (np.abs(vb) + 1e-6) * np.sign(vb + 1e-12)
        out[f"{b}/{a}"] = vb / (np.abs(va) + 1e-6) * np.sign(va + 1e-12)
        out[f"{a}*{b}^2"], out[f"{a}^2*{b}"] = va * vb * vb, va * va * vb
    P = pd.DataFrame(out, index=D.index)
    return P.clip(-1e30, 1e30).astype(np.float32)


def fit_predict(Xtr_, ytr_, Xva_, seed):
    a, b = train_test_split(np.arange(len(ytr_)), test_size=0.1, stratify=ytr_, random_state=seed)
    m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    m.fit(Xtr_.iloc[a], ytr_[a], eval_set=[(Xtr_.iloc[b], ytr_[b])], eval_metric="auc",
          callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict_proba(Xva_)[:, 1]


arms = ("F only", "F+raw", "F+raw+inter")
oof = {a: np.zeros(len(y)) for a in arms}
kept_by_fold = []
for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(POOL, y)):
    top = gain_rank(POOL.iloc[tr], y[tr], f).index[: args.k].tolist()
    P = make_pairs(POOL, top)
    imp = gain_rank(pd.concat([POOL.iloc[tr], P.iloc[tr]], axis=1), y[tr], 100 + f)
    keep = [c for c in imp.index if c in P.columns][: args.keep]
    kept_by_fold.append(keep)
    XI = pd.concat([POOL, P[keep]], axis=1)
    del P
    for s in range(args.seeds):
        sd = 100 * f + s
        oof["F only"][va] += fit_predict(F.iloc[tr], y[tr], F.iloc[va], sd) / args.seeds
        oof["F+raw"][va] += fit_predict(POOL.iloc[tr], y[tr], POOL.iloc[va], sd) / args.seeds
        oof["F+raw+inter"][va] += fit_predict(XI.iloc[tr], y[tr], XI.iloc[va], sd) / args.seeds
    print(f"fold {f}: " + "  ".join(f"{a} {roc_auc_score(y[va], oof[a][va]):.4f}" for a in arms) + f"  top kept {keep[:2]}", flush=True)

rng = np.random.default_rng(0)
idx = [rng.integers(0, len(y), len(y)) for _ in range(400)]
print()
for a in arms:
    d = [roc_auc_score(y[i], oof[a][i]) - roc_auc_score(y[i], oof["F only"][i]) for i in idx]
    print(f"{a:12s} OOF {roc_auc_score(y, oof[a]):.4f}  vs F only CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
cnt = Counter(c for k in kept_by_fold for c in k)
stable = [c for c, n in cnt.items() if n >= 4]
print(f"interactions kept in >=4 of 5 folds: {len(stable)}")
raw_inv = [c for c in stable if any(t in c for t in ("RAW_", "CUMSUM_", "DIFF_"))]
print(f"  of which involve raw-scale pool columns: {len(raw_inv)} -> {raw_inv[:8]}")
json.dump({"kept_by_fold": kept_by_fold, "stable": stable}, open("outputs/interactions2.json", "w"))
np.save("outputs/oof_inter2.npy", np.vstack([oof[a] for a in arms]))
