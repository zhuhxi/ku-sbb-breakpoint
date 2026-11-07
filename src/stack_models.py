"""Second and third model types on the combo10gnj columns, same folds as confirm.py / train.run_cv
(StratifiedKFold(5, shuffle, 42), early stopping on an inner stratified 10%), for blending with the LightGBM.

    python stack_models.py xgb      # -> outputs/factory/stack_xgb.npz (oof, test in sample-submission order)
    python stack_models.py cat      # -> outputs/factory/stack_cat.npz
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

kind = sys.argv[1]
NJ = 28
t0 = time.time()
FAMS = ["st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost", "genA", "genB", "gnoise", "gjlr"]   # hpc/gnj.sbatch
PAT = r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_"
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
ld = lambda p, idx: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
keep = json.load(open("outputs/harvest_keep.json"))


def frame(idx, part):
    N = pd.concat([ld(f"outputs/factory/{f}.parquet", idx) for f in FAMS], axis=1)
    N = N.loc[:, N.columns.str.contains(PAT, regex=True)]
    return pd.concat([ld(f"outputs/feat_{part}.parquet", idx), ld(f"outputs/magic_b_{part}.parquet", idx),
                      ld(f"outputs/harvest_{part}.parquet", idx)[keep], N.loc[:, ~N.columns.duplicated()]], axis=1)


X, Xt = frame(lab.index, "train"), frame(sub["id"], "test")
Xt = Xt[X.columns]
print(kind, X.shape, flush=True)


def fit(Xa, ya, Xb, yb, seed):
    if kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.02, max_depth=6, min_child_weight=5, subsample=0.8,
                              colsample_bytree=0.5, reg_lambda=1.0, tree_method="hist", eval_metric="auc",
                              early_stopping_rounds=300, n_jobs=NJ, random_state=seed)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        return m, m.best_iteration
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=5000, learning_rate=0.04, depth=6, l2_leaf_reg=3, rsm=0.5, eval_metric="AUC",
                           od_type="Iter", od_wait=300, thread_count=NJ, random_seed=seed, verbose=False, allow_writing_files=False)
    m.fit(Xa, ya, eval_set=(Xb, yb), use_best_model=True)
    return m, m.get_best_iteration()


oof, pred = np.zeros(len(y)), np.zeros(len(Xt))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=100 * k)   # train.fit_lgb inner split, seed 0
    m, it = fit(X.iloc[a], y[a], X.iloc[b], y[b], 100 * k)
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    pred += m.predict_proba(Xt)[:, 1] / 5
    print(f"fold {k}: {kind} {roc_auc_score(y[va], oof[va]):.4f} (iters {it})  {time.time() - t0:.0f}s", flush=True)
np.savez(f"outputs/factory/stack_{kind}.npz", oof=oof, test=pred)
print(f"OOF {kind} {roc_auc_score(y, oof):.4f}  {time.time() - t0:.0f}s")
