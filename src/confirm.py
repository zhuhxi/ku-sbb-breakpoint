"""Confirm a factory selection at production settings and write test predictions.

    python confirm.py st_ABSD_RES_SQ fe ar2 --pat "^fe_|^st_|_(ab|ba)_" --tag combo1 [--seeds 3]

Base (features.py + magic B + 58 harvest columns) + the selected factory columns, train.run_cv
(LightGBM lr 0.02, early stopping on an inner 10%, StratifiedKFold(5, shuffle, 42)). Reports OOF and
test-like AUC with a paired bootstrap against the current submission's OOF (brute/oof_lgb_inter_B.npy),
and rank blends with it. Saves outputs/factory/confirm_{tag}.npz (oof, test in sample-submission order).
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from train import LGB_PARAMS, run_cv

ap = argparse.ArgumentParser()
ap.add_argument("fams", nargs="*")
ap.add_argument("--pat", default=None)
ap.add_argument("--tag", required=True)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--select-k", type=int, default=None)   # per-fold top-k feature selection
ap.add_argument("--colsample", type=float, default=None)   # override LGB colsample_bytree
args = ap.parse_args()

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
ld = lambda p, idx: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
keep = json.load(open("outputs/harvest_keep.json"))


def frame(idx, part):
    F = ld(f"outputs/feat_{part}.parquet", idx)
    B = ld(f"outputs/magic_b_{part}.parquet", idx)
    H = ld(f"outputs/harvest_{part}.parquet", idx)[keep]
    N = pd.concat([ld(f"outputs/factory/{f}.parquet", idx) for f in args.fams], axis=1) if args.fams else pd.DataFrame(index=idx)
    if args.pat:
        N = N.loc[:, N.columns.str.contains(args.pat, regex=True)]
    return pd.concat([F, B, H, N.loc[:, ~N.columns.duplicated()]], axis=1)


X, Xt = frame(lab.index, "train"), frame(sub["id"], "test")
Xt = Xt[X.columns]
print(f"{args.tag}: {X.shape[1]} columns", flush=True)
oof, pred, _ = run_cv(X, y, Xt, seeds=range(args.seeds), params={**LGB_PARAMS, "n_jobs": int(__import__("os").environ.get("CONF_JOBS", 30)), **({"colsample_bytree": args.colsample} if args.colsample else {})}, logreg=False, verbose=True, select_k=args.select_k)
o, p = oof["lgb"], pred["lgb"]
np.savez(f"outputs/factory/confirm_{args.tag}.npz", oof=o, test=p)

flag = X["mb_rescaled"].fillna(0).to_numpy() == 1
pt = Xt["mb_rescaled"].fillna(0).mean()
w_mix = np.where(flag, pt / flag.mean(), (1 - pt) / (1 - flag.mean()))
ref = np.load("outputs/brute/oof_lgb_inter_B.npy").mean(0)
A = lambda s, w=None: roc_auc_score(y, s, sample_weight=w)
print(f"OOF {A(o):.4f}  test-like {A(o, w_mix):.4f}   | inter_B OOF {A(ref):.4f}  test-like {A(ref, w_mix):.4f}")
rng = np.random.default_rng(0)
d = []
for _ in range(500):
    i = rng.integers(0, len(y), len(y))
    d.append(roc_auc_score(y[i], o[i]) - roc_auc_score(y[i], ref[i]))
print(f"vs inter_B: {A(o) - A(ref):+.4f}  CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
for w in (0.3, 0.5, 0.7):
    b = w * rankdata(o) + (1 - w) * rankdata(ref)
    print(f"  rank blend w_new={w}: OOF {A(b):.4f}  test-like {A(b, w_mix):.4f}")
