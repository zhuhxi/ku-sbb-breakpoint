"""Fast paired screen of factory families against the current base set.

    python screen.py st                 # base vs base + outputs/factory/st.parquet
    python screen.py st cost --tag b1   # several families in one arm
    python screen.py st --cols st_keep.json   # only listed columns
    python screen.py ar2 --pat "_(ab|ba)_" --tag ar2_pred   # only columns matching a regex

Base = features.py (F) + magic B + the 58 adopted harvest columns. LightGBM lr 0.05, 5 folds
(StratifiedKFold(5, shuffle, 42)), 2 seeds, early stopping on an inner 10% split; the base OOF is cached so
each screen trains only the new arm. Prints OOF AUC, test-like AUC (rescaled family at its test share),
the paired bootstrap delta and the top new columns by gain; appends one line to outputs/factory/screen.log.
"""
import argparse
import os
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("fams", nargs="*")
ap.add_argument("--cols", default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--pat", default=None, help="regex: keep only matching new columns")
ap.add_argument("--extra", default=None, help="parquet of extra columns indexed by id")
ap.add_argument("--seed-offset", type=int, default=0, help="shift model seeds (noise-floor check)")
ap.add_argument("--keep", default=None, help="json list: restrict the WHOLE frame (base + new) to these columns")
ap.add_argument("--base", default=None, choices=[None, "combo10"], help="screen on top of a larger base (own OOF cache)")
args = ap.parse_args()

P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, n_estimators=3000, verbose=-1, n_jobs=int(os.environ.get("NTHREADS", 30)))
OUT = Path("outputs/factory")
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.to_numpy()
ld = lambda p: pd.read_parquet(p).reindex(lab.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def base_frame():
    F, B, H = ld("outputs/feat_train.parquet"), ld("outputs/magic_b_train.parquet"), ld("outputs/harvest_train.parquet")
    X = pd.concat([F, B, H[json.load(open("outputs/harvest_keep.json"))]], axis=1)
    if args.base == "combo10":
        N = pd.concat([ld(OUT / f"{f}.parquet") for f in ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")], axis=1)
        N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$", regex=True)]
        X = pd.concat([X, N.loc[:, ~N.columns.duplicated()]], axis=1)
    return X


def cv(X, seeds):
    oof, imp = np.zeros(len(y)), pd.Series(0.0, index=X.columns)
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        for s in range(seeds):
            a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=100 * k + s)
            m = lgb.LGBMClassifier(**P, random_state=100 * k + s + args.seed_offset)
            m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[va] += m.predict_proba(X.iloc[va])[:, 1] / seeds
            imp += m.booster_.feature_importance("gain")
    return oof, imp


if __name__ == "__main__":
    t = time.time()
    X0 = base_frame()
    flag = X0["mb_rescaled"].fillna(0).to_numpy() == 1
    pt = pd.read_parquet("outputs/magic_b_test.parquet")["mb_rescaled"].fillna(0).mean()
    w_mix = np.where(flag, pt / flag.mean(), (1 - pt) / (1 - flag.mean()))
    cache = OUT / f"base_oof_s{args.seeds}{'_' + args.base if args.base else ''}.npy"
    if cache.exists():
        o0 = np.load(cache)
    else:
        o0, _ = cv(X0, args.seeds)
        np.save(cache, o0)
        print(f"base {X0.shape[1]} cols  OOF {roc_auc_score(y, o0):.4f}", flush=True)
    parts = [ld(OUT / f"{f}.parquet") for f in args.fams]
    if args.extra:
        parts.append(ld(args.extra))
    N = pd.concat(parts, axis=1)
    if args.cols:
        N = N[[c for c in json.load(open(args.cols)) if c in N]]
    if args.pat:
        N = N.loc[:, N.columns.str.contains(args.pat, regex=True)]
    N = N.loc[:, ~N.columns.duplicated() & ~N.columns.isin(X0.columns)]
    X = pd.concat([X0, N], axis=1)
    if args.keep:
        X = X[[c for c in json.load(open(args.keep)) if c in X]]
        N = X.loc[:, ~X.columns.isin(X0.columns)]
        print(f"keep list: {X.shape[1]} columns ({N.shape[1]} new)", flush=True)
    o1, imp = cv(X, args.seeds)
    tag = args.tag or "+".join(args.fams)
    np.save(OUT / f"oof_{tag}.npy", o1)
    imp[N.columns].sort_values(ascending=False).to_csv(OUT / f"imp_{tag}.csv")
    rng = np.random.default_rng(0)
    d = []
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[i], o1[i]) - roc_auc_score(y[i], o0[i]))
    a0, a1 = roc_auc_score(y, o0), roc_auc_score(y, o1)
    m0, m1 = roc_auc_score(y, o0, sample_weight=w_mix), roc_auc_score(y, o1, sample_weight=w_mix)
    tag = f"{args.base}:{tag}" if args.base else tag
    line = (f"{tag:28s} +{N.shape[1]:5d} cols  OOF {a0:.4f} -> {a1:.4f} ({a1 - a0:+.4f}, CI [{np.percentile(d, 2.5):+.4f}, "
            f"{np.percentile(d, 97.5):+.4f}])  test-like {m1 - m0:+.4f}  {time.time() - t:.0f}s")
    print(line)
    print("  top new:", list(imp[N.columns].sort_values(ascending=False).index[:12]))
    with open(OUT / "screen.log", "a") as f:
        f.write(line + "\n")
