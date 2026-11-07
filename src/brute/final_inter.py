"""Fit the top-162 -> 91k-interaction LightGBM on all training series and build a blended submission.

    python brute/final_inter.py [--extra outputs/magic_b --extra_cols outputs/magic_b_keep.json] [--tag _B]

Same recipe as lgb_inter.py, fitted once on the full training set (inner 10% only for early
stopping): all-91k model and its top-800 retrain, 3 seeds each. Test predictions are rank-blended
with equal weights with the current ensemble (outputs/submission_ens.csv).
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser()
ap.add_argument("--extra", default=None, help="prefix, e.g. outputs/magic_b -> _train/_test.parquet")
ap.add_argument("--extra_cols", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()
from inter_lib import Inter, P  # noqa: E402  (the exact builder and params used in lgb_inter.py)

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()


def pool(split, idx):
    ld = lambda p: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    parts = [ld(f"outputs/feat_{split}.parquet"), ld(f"outputs/raw_{split}.parquet")]
    if args.extra:
        E = ld(f"{args.extra}_{split}.parquet")
        parts.append(E[json.load(open(args.extra_cols))] if args.extra_cols else E)
    return pd.concat(parts, axis=1)


PT, PS = pool("train", lab.index), pool("test", sub["id"])
PS = PS[PT.columns]
rk = lgb.train({**P, "feature_fraction": 0.3, "seed": 0}, lgb.Dataset(PT, y), 600)
top = pd.Series(rk.feature_importance("gain"), PT.columns).nlargest(162).index.tolist()
T, Ts = PT[top].to_numpy(), PS[top].to_numpy()
print(f"pool {PT.shape[1]} -> top 162; extra columns in top 162: {[c for c in top if c.startswith('mb_')]}", flush=True)

a, b = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=0)
I = Inter(T)
cols_all = list(range(len(top))) + I.names
Ma = I.full(T[a]); da = lgb.Dataset(Ma, y[a], params={"max_bin": 63}, free_raw_data=True).construct(); del Ma; gc.collect()
Mb = I.full(T[b]); db = lgb.Dataset(Mb, y[b], reference=da).construct(); del Mb; gc.collect()
Ms = I.full(Ts)

p91, p800 = np.zeros(len(Ts)), np.zeros(len(Ts))
for s in range(args.seeds):
    m1 = lgb.train({**P, "feature_fraction": 0.1, "seed": s}, da, 5000, valid_sets=[db],
                   callbacks=[lgb.early_stopping(300, verbose=False)])
    p91 += m1.predict(Ms, num_iteration=m1.best_iteration) / args.seeds
    g = pd.Series(m1.feature_importance("gain"), range(len(cols_all)))
    keep = [cols_all[i] for i in g.nlargest(800).index]
    Sa = lgb.Dataset(I.select(T[a], keep), y[a], params={"max_bin": 255})
    Sb = lgb.Dataset(I.select(T[b], keep), y[b], reference=Sa)
    m2 = lgb.train({**P, "feature_fraction": 0.5, "seed": s}, Sa, 5000, valid_sets=[Sb],
                   callbacks=[lgb.early_stopping(300, verbose=False)])
    p800 += m2.predict(I.select(Ts, keep), num_iteration=m2.best_iteration) / args.seeds
    print(f"seed {s}: all-91k {m1.best_iteration} trees (inner AUC {m1.best_score['valid_0']['auc']:.4f}), "
          f"top-800 {m2.best_iteration} trees (inner AUC {m2.best_score['valid_0']['auc']:.4f})", flush=True)

ens = pd.read_csv("outputs/submission_ens.csv", dtype={"id": str}).set_index("id").loc[sub["id"], "break_probability"].to_numpy()
r = lambda v: rankdata(v) / len(v)
out = sub.copy()
out["break_probability"] = (r(p91) + r(p800) + r(ens)) / 3
assert list(out.columns) == ["id", "break_probability"] and out["id"].equals(sub["id"])
assert np.isfinite(out["break_probability"]).all() and out["break_probability"].between(0, 1).all()
path = f"outputs/submission_inter{args.tag}.csv"
out.to_csv(path, index=False)
np.save(f"outputs/brute/test_pred_inter{args.tag}.npy", np.vstack([p91, p800]))
print(f"wrote {path}  rank corr with old ensemble {np.corrcoef(r(out['break_probability']), r(ens))[0, 1]:.3f}")
