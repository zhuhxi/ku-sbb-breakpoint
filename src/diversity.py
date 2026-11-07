"""XGBoost / CatBoost (GPU) on the combo4 feature set, for a three-library blend.

    python diversity.py --model xgb --gpu 0
    python diversity.py --model cat --gpu 1

Same columns as confirm.py combo4 (features.py + magic B + 58 harvest + st_RES10/st_ARES10/fe/ar2 pred/cost gain),
StratifiedKFold(5, shuffle, 42), early stopping on an inner 10% split, 2 seeds. Saves
outputs/factory/div_{model}.npz (oof, test in sample-submission order).
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["xgb", "cat"], required=True)
ap.add_argument("--gpu", default="0")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--fams", default="st_RES10,st_ARES10,fe,ar2,cost")
ap.add_argument("--pat", default=r"^fe_|^st_|_(?:ab|ba)_|_gain$")
ap.add_argument("--tag", default="")
args = ap.parse_args()

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
ld = lambda p, idx: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
keep = json.load(open("outputs/harvest_keep.json"))


def frame(idx, part):
    N = pd.concat([ld(f"outputs/factory/{f}.parquet", idx) for f in args.fams.split(",")], axis=1)
    N = N.loc[:, N.columns.str.contains(args.pat, regex=True)]
    return pd.concat([ld(f"outputs/feat_{part}.parquet", idx), ld(f"outputs/magic_b_{part}.parquet", idx),
                      ld(f"outputs/harvest_{part}.parquet", idx)[keep], N], axis=1)


X, Xt = frame(lab.index, "train"), frame(sub["id"], "test")
Xt = Xt[X.columns]
X, Xt = X.to_numpy(), Xt.to_numpy()


def fit_predict(Xa, ya, Xb, yb, Xs, seed):
    if args.model == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.02, max_depth=6, min_child_weight=5, subsample=0.8,
                              colsample_bytree=0.5, reg_lambda=1.0, tree_method="hist", device=f"cuda:{args.gpu}",
                              eval_metric="auc", early_stopping_rounds=300, random_state=seed, verbosity=0)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        return [m.predict_proba(s)[:, 1] for s in Xs]
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=6000, learning_rate=0.03, depth=7, l2_leaf_reg=3, task_type="GPU", devices=args.gpu,
                           eval_metric="AUC", od_type="Iter", od_wait=300, random_seed=seed, verbose=False)
    m.fit(Xa, ya, eval_set=(Xb, yb), use_best_model=True)
    return [m.predict_proba(s)[:, 1] for s in Xs]


oof, pred = np.zeros(len(y)), np.zeros(len(Xt))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    for s in range(args.seeds):
        a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=100 * k + s)
        pv, pt = fit_predict(X[a], y[a], X[b], y[b], [X[va], Xt], 100 * k + s)
        oof[va] += pv / args.seeds
        pred += pt / (5 * args.seeds)
    print(f"{args.model} fold {k}: {roc_auc_score(y[va], oof[va]):.4f}", flush=True)
print(f"{args.model} OOF AUC {roc_auc_score(y, oof):.4f}")
np.savez(f"outputs/factory/div_{args.model}{args.tag}.npz", oof=oof, test=pred)
