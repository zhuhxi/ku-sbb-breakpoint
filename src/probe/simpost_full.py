"""Full-feature generator posterior: LightGBM trained ONLY on the clean-generator fakes (outputs/aug/X_simku.parquet from
aug_features.py simku), on the real model's columns that the fakes have, float-bit (mb_) columns dropped (they are
code artefacts, not statistics). Scores real train + test -> outputs/factory/simpost_full.parquet; prints TSTR.

    python probe/simpost_full.py
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

t0 = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
FAM = ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")


def frame(idx, part):
    ld = lambda p: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    N = pd.concat([ld(f"outputs/factory/{f}.parquet") for f in FAM], axis=1)
    N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$", regex=True)]
    X = pd.concat([ld(f"outputs/feat_{part}.parquet"), ld(f"outputs/harvest_{part}.parquet")[json.load(open("outputs/harvest_keep.json"))],
                   N.loc[:, ~N.columns.duplicated()]], axis=1)
    return X.loc[:, ~X.columns.duplicated()]


XS = pd.read_parquet("outputs/aug/X_simku.parquet")
M = pd.read_parquet("outputs/aug/meta_simku.parquet")
XR, XT = frame(lab.index, "train"), frame(sub["id"], "test")
cols = [c for c in XR.columns if c in XS.columns and not c.startswith("mb_")]
XS, XR, XT = XS[cols].replace([np.inf, -np.inf], np.nan), XR[cols], XT[cols]
ys = M.loc[XS.index, "label"].to_numpy()
y = lab.to_numpy()
print(f"{len(cols)} columns, {len(XS)} fakes, load {time.time() - t0:.0f}s", flush=True)
prm = dict(objective="binary", learning_rate=.05, num_leaves=31, min_child_samples=40, subsample=.8, subsample_freq=1,
           colsample_bytree=.5, reg_lambda=1., n_estimators=3000, verbose=-1, n_jobs=-1)
a, b = train_test_split(np.arange(len(ys)), test_size=.1, stratify=ys, random_state=0)
m = lgb.LGBMClassifier(**prm).fit(XS.iloc[a], ys[a], eval_set=[(XS.iloc[b], ys[b])], eval_metric="auc",
                                  callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"fake holdout AUC {roc_auc_score(ys[b], m.predict_proba(XS.iloc[b])[:, 1]):.4f}  trees {m.best_iteration_}", flush=True)
pr, pt = m.predict_proba(XR)[:, 1], m.predict_proba(XT)[:, 1]
print(f"TSTR (fakes only -> real train labels) AUC {roc_auc_score(y, pr):.4f}", flush=True)
q = dict(prm, learning_rate=.1, n_estimators=400)
o = cross_val_predict(lgb.LGBMClassifier(**q), XR, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
print(f"real-trained 5-fold on the same columns (lr .1, 400 trees) AUC {roc_auc_score(y, o):.4f}")
pd.DataFrame({"simf_post": np.r_[pr, pt]}, index=list(lab.index) + list(sub["id"])).to_parquet("outputs/factory/simpost_full.parquet")
print(f"done {time.time() - t0:.0f}s")
