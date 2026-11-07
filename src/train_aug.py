"""Single combo10 LGB trained with augmented rows; scored on the ORIGINAL rows only.

    python train_aug.py --aug crop spike --tag both

Same folds, seeds and parameters as screen.py --base combo10 (so the baseline OOF cache is comparable).
Augmented rows go to the fold of their parent: in outer fold k they are used for training only when the parent
is in the training part, and never when the parent is in the inner early-stopping split. Prints OOF vs the
baseline with a paired bootstrap CI, test-like AUC, and AUC on the two weak groups found by the error analysis
(shorter segment <= 150, spike > 8 MAD after the boundary). Writes outputs/aug/oof_{tag}.npy, test_{tag}.npy.
"""
import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--aug", nargs="*", default=[])
ap.add_argument("--tag", required=True)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
ap.add_argument("--w", type=float, default=1.0, help="sample weight of augmented rows")
args = ap.parse_args()

P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, n_estimators=3000, verbose=-1, n_jobs=int(os.environ.get("NTHREADS", 30)))
OUT, FAC = Path("outputs/aug"), Path("outputs/factory")
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()


def frame(idx, split):
    ld = lambda p: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    F, B = ld(f"outputs/feat_{split}.parquet"), ld(f"outputs/magic_b_{split}.parquet")
    H = ld(f"outputs/harvest_{split}.parquet")[json.load(open("outputs/harvest_keep.json"))]
    N = pd.concat([ld(FAC / f"{f}.parquet") for f in ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")], axis=1)
    N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$", regex=True)]
    X = pd.concat([F, B, H, N.loc[:, ~N.columns.duplicated()]], axis=1)
    return X.loc[:, ~X.columns.duplicated()]


t = time.time()
X = frame(lab.index, "train")
Xt = frame(sub["id"], "test")
meta = pd.concat([pd.read_parquet(OUT / "meta_all.parquet")] + ([pd.read_parquet(OUT / "meta_sim.parquet")] if (OUT / "meta_sim.parquet").exists() else []))
A = [pd.read_parquet(OUT / f"X_{k}.parquet")[X.columns] for k in args.aug]
XA = pd.concat(A) if A else X.iloc[:0]
mA = meta.reindex(XA.index)
yA = mA["label"].to_numpy()
pos = pd.Series(np.arange(len(y)), index=lab.index)
parA = pos.reindex(mA["parent"]).fillna(-1).astype(int).to_numpy() if len(XA) else np.zeros(0, int)   # -1: generated, no parent
print(f"{X.shape[1]} cols, {len(XA)} augmented rows, load {time.time() - t:.0f}s", flush=True)

oof, test = np.zeros(len(y)), np.zeros(len(Xt))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    if k not in args.folds:
        continue
    for s in range(args.seeds):
        a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=100 * k + s)
        use = np.isin(parA, a) | (parA < 0)
        Xa = pd.concat([X.iloc[a], XA[use]])
        ya = np.r_[y[a], yA[use]]
        wa = np.r_[np.ones(len(a)), np.full(use.sum(), args.w)]
        m = lgb.LGBMClassifier(**P, random_state=100 * k + s)
        m.fit(Xa, ya, sample_weight=wa, eval_set=[(X.iloc[b], y[b])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] += m.predict_proba(X.iloc[va])[:, 1] / args.seeds
        test += m.predict_proba(Xt)[:, 1] / (len(args.folds) * args.seeds)
    print(f"fold {k}: AUC {roc_auc_score(y[va], oof[va]):.4f}  train rows {len(ya)}  {time.time() - t:.0f}s", flush=True)

np.save(OUT / f"oof_{args.tag}.npy", oof)
np.save(OUT / f"test_{args.tag}.npy", test)
o0 = np.load(FAC / "base_oof_s2_combo10.npy")
ev = np.concatenate([va for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)) if k in args.folds])
y, oof, o0, X = y[ev], oof[ev], o0[ev], X.iloc[ev]
rng = np.random.default_rng(0)
d = []
for _ in range(300):
    i = rng.integers(0, len(y), len(y))
    d.append(roc_auc_score(y[i], oof[i]) - roc_auc_score(y[i], o0[i]))
flag = X["mb_rescaled"].fillna(0).to_numpy() == 1
pt = pd.read_parquet("outputs/magic_b_test.parquet")["mb_rescaled"].fillna(0).mean()
w_mix = np.where(flag, pt / flag.mean(), (1 - pt) / (1 - flag.mean()))
a0, a1 = roc_auc_score(y, o0), roc_auc_score(y, oof)
m0, m1 = roc_auc_score(y, o0, sample_weight=w_mix), roc_auc_score(y, oof, sample_weight=w_mix)
hc = pd.read_parquet("outputs/probe/hard_cases.parquet").reindex(lab.index[ev])
hc2 = pd.read_parquet("outputs/probe/hard_cases2.parquet").reindex(lab.index[ev])
grp = ""
for nm, msk in (("short<=150", hc["nmin"].to_numpy() <= 150), ("spike>8", hc2["maxsp_a"].to_numpy() > 8)):
    grp += f"  {nm} {roc_auc_score(y[msk], o0[msk]):.4f}->{roc_auc_score(y[msk], oof[msk]):.4f}"
line = (f"aug:{args.tag:12s} w={args.w:g} OOF {a0:.4f} -> {a1:.4f} ({a1 - a0:+.4f}, CI [{np.percentile(d, 2.5):+.4f}, "
        f"{np.percentile(d, 97.5):+.4f}])  test-like {m1 - m0:+.4f}{grp}  {time.time() - t:.0f}s")
print(line)
with open(OUT / "aug.log", "a") as f:
    f.write(line + "\n")
