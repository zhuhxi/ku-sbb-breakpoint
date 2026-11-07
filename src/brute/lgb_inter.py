"""datatech-style interactions trained all at once by LightGBM, leak-free inside each fold.

    python brute/lgb_inter.py [--smoke] [--top 162] [--seeds 2]

Per outer fold (StratifiedKFold(5, shuffle, 42), same as train.py), training rows only:
  1. rank the 350 pool columns (features.py + rawstats.py) by LightGBM gain, keep the top 162;
  2. every pair x {a*b, a+b, a-b, a/b, b/a, a*b^2, a^2*b} + each squared = 91,449 interactions;
  3. arm "all 91k": LightGBM on 162 + 91,449 columns at once;
  4. arm "top 800": keep the 800 highest-gain columns of that model, retrain (datatech's next step).
Arm "base" = the 242 features.py columns. All arms share folds, inner split and seeds.
Interaction blocks are built per row subset and binned immediately to keep RAM under ~25 GB.
"""
import argparse
import json
import os
import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--top", type=int, default=162)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--extra", default=None, help="parquet of extra pool columns, e.g. outputs/magic_b_train.parquet")
ap.add_argument("--extra_cols", default=None, help="json list restricting the extra columns")
ap.add_argument("--tag", default="", help="suffix for the saved OOF file")
ap.add_argument("--seed-offset", type=int, default=0, help="shift the model seeds (inner split unchanged)")
args = ap.parse_args()

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.to_numpy()
load = lambda p: pd.read_parquet(p).reindex(lab.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
F = load("outputs/feat_train.parquet")
POOL = pd.concat([F, load("outputs/raw_train.parquet")], axis=1)
if args.extra:
    E = load(args.extra)
    if args.extra_cols:
        E = E[json.load(open(args.extra_cols))]
    POOL = pd.concat([POOL, E], axis=1)
    print(f"pool {POOL.shape[1]} columns incl. {E.shape[1]} extra from {args.extra}", flush=True)
# test pool with the same columns: fold models predict it too (fold-averaged test preds for stacking)
sub_ids = pd.read_csv("data/sample_submission.csv", dtype={"id": str})["id"]
lt = lambda p: pd.read_parquet(p).reindex(sub_ids).replace([np.inf, -np.inf], np.nan).astype(np.float32)
POOL_T = pd.concat([lt("outputs/feat_test.parquet"), lt("outputs/raw_test.parquet")]
                   + ([lt(args.extra.replace("_train", "_test"))] if args.extra else []), axis=1)[POOL.columns]
F_T = POOL_T[F.columns]
from inter_lib import OPS, Inter, P  # noqa: E402,F401  (shared with final_inter.py)


def train(da, db, seed, colsample):
    return lgb.train({**P, "feature_fraction": colsample, "seed": seed}, da, 5000, valid_sets=[db],
                     callbacks=[lgb.early_stopping(300, verbose=False)])


folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(POOL, y))
if args.smoke:
    folds = folds[:1]
arms = ("base", "all 91k", "top 800")
oof = {k: np.zeros(len(y)) for k in arms}
tst = {k: np.zeros(len(F_T)) for k in arms}
for f, (tr, va) in enumerate(folds):
    t = time.time()
    rk = lgb.train({**P, "feature_fraction": 0.3, "seed": f}, lgb.Dataset(POOL.iloc[tr], y[tr]), 600)
    top = pd.Series(rk.feature_importance("gain"), POOL.columns).nlargest(args.top).index.tolist()
    T = POOL[top].to_numpy()
    a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=f)
    I = Inter(T[tr])
    cols_all = list(range(len(top))) + I.names
    Ma = I.full(T[a]); da = lgb.Dataset(Ma, y[a], params={"max_bin": 63}, free_raw_data=True).construct(); del Ma; gc.collect()
    Mb = I.full(T[b]); db = lgb.Dataset(Mb, y[b], reference=da).construct(); del Mb; gc.collect()
    Mv = I.full(T[va])
    Tt = POOL_T[top].to_numpy()
    Mt = I.full(Tt)
    w = 1 / (len(folds) * args.seeds)
    print(f"fold {f}: top {len(top)} -> {len(cols_all):,} columns binned in {time.time() - t:.0f}s", flush=True)
    Fa, Fb = lgb.Dataset(F.iloc[a].to_numpy(), y[a], params={"max_bin": 255}), None
    Fb = lgb.Dataset(F.iloc[b].to_numpy(), y[b], reference=Fa)
    for s in range(args.seeds):
        sd = 100 * f + s + args.seed_offset
        m0 = train(Fa, Fb, sd, 0.5)
        oof["base"][va] += m0.predict(F.iloc[va].to_numpy(), num_iteration=m0.best_iteration) / args.seeds
        tst["base"] += m0.predict(F_T.to_numpy(), num_iteration=m0.best_iteration) * w
        t = time.time()
        m1 = train(da, db, sd, 0.1)
        oof["all 91k"][va] += m1.predict(Mv, num_iteration=m1.best_iteration) / args.seeds
        tst["all 91k"] += m1.predict(Mt, num_iteration=m1.best_iteration) * w
        g = pd.Series(m1.feature_importance("gain"), range(len(cols_all)))
        keep = [cols_all[i] for i in g.nlargest(800).index]
        Sa, Sb = lgb.Dataset(I.select(T[a], keep), y[a], params={"max_bin": 255}), None
        Sb = lgb.Dataset(I.select(T[b], keep), y[b], reference=Sa)
        m2 = train(Sa, Sb, sd, 0.5)
        oof["top 800"][va] += m2.predict(I.select(T[va], keep), num_iteration=m2.best_iteration) / args.seeds
        tst["top 800"] += m2.predict(I.select(Tt, keep), num_iteration=m2.best_iteration) * w
        if os.environ.get("SAVE"):   # checkpoint for CPU inference of the "top 800" arm
            d = f"{os.environ['SAVE']}/inter{args.tag}/fold{f}"; os.makedirs(d, exist_ok=True)
            json.dump(top, open(f"{d}/top.json", "w")); np.savez(f"{d}/inter.npz", med=I.med, eps=I.eps)
            json.dump([int(c) if isinstance(c, (int, np.integer)) else c for c in keep], open(f"{d}/keep_seed{s}.json", "w"))
            m2.save_model(f"{d}/top800_seed{s}.txt", num_iteration=m2.best_iteration)
        pretty = lambda c: top[c] if isinstance(c, (int, np.integer)) else (
            "sq(" + top[int(c.split("__")[1])] + ")" if c.startswith("sq__") else
            f"{c.split('__')[0]}({top[int(c.split('__')[1])]}, {top[int(c.split('__')[2])]})")
        n_int = sum(not isinstance(c, (int, np.integer)) for c in keep)
        print(f"  seed {s}: all-91k {m1.best_iteration} trees [{time.time() - t:.0f}s]; top 800 = {n_int} interactions + "
              f"{800 - n_int} base; top gain: {[pretty(cols_all[i]) for i in g.nlargest(3).index]}", flush=True)
    del da, db, Mv, Mt
    gc.collect()
    print(f"fold {f}: " + "  ".join(f"{k} {roc_auc_score(y[va], oof[k][va]):.4f}" for k in arms), flush=True)

if not args.smoke:
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(400)]
    for k in arms:
        d = [roc_auc_score(y[i], oof[k][i]) - roc_auc_score(y[i], oof["base"][i]) for i in idx]
        print(f"{k:8s} OOF AUC {roc_auc_score(y, oof[k]):.4f}  vs base CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
    np.save(f"outputs/brute/oof_lgb_inter{args.tag}.npy", np.vstack([oof[k] for k in arms]))
    np.save(f"outputs/brute/test_lgb_inter{args.tag}.npy", np.vstack([tst[k] for k in arms]))
