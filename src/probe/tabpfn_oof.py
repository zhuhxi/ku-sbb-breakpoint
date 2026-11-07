"""TabPFN 3.5 out-of-fold + test predictions on the combo10gnj columns (888), the setting that did best on fold 0
(probe/tabpfn_expert.py 1 0 8: all columns, all train rows of the fold as context, n_estimators 8, median fill).
Folds = StratifiedKFold(5, shuffle, 42). Each fold's model also predicts the test set; test = mean over folds.

    TABPFN_TOKEN=... CUDA_VISIBLE_DEVICES=0 python probe/tabpfn_oof.py 0 2 4    # folds to run -> outputs/factory/tabpfn_f{k}.npz
    python probe/tabpfn_oof.py merge                                            # -> outputs/factory/tabpfn_oof.npz (oof, test)
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
FAM = ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost", "genA", "genB")


def frame(idx, part):
    ld = lambda p: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    N = pd.concat([ld(f"outputs/factory/{f}.parquet") for f in FAM], axis=1)
    N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_", regex=True)]
    X = pd.concat([ld(f"outputs/feat_{part}.parquet"), ld(f"outputs/magic_b_{part}.parquet"),
                   ld(f"outputs/harvest_{part}.parquet")[json.load(open("outputs/harvest_keep.json"))],
                   N.loc[:, ~N.columns.duplicated()], ld("outputs/factory/gnoise.parquet"), ld("outputs/factory/gjlr.parquet")], axis=1)
    return X.loc[:, ~X.columns.duplicated() & ~X.columns.str.startswith("__")]


folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
if sys.argv[1] == "merge":
    oof, test = np.zeros(len(y)), np.zeros(len(sub))
    for k, (tr, va) in enumerate(folds):
        z = np.load(f"outputs/factory/tabpfn_f{k}.npz")
        oof[va] = z["va"]
        test += z["test"] / 5
        print(f"fold {k}: AUC {roc_auc_score(y[va], z['va']):.4f}")
    print(f"TabPFN OOF AUC {roc_auc_score(y, oof):.4f}")
    np.savez("outputs/factory/tabpfn_oof.npz", oof=oof, test=test)
    sys.exit()

from tabpfn import TabPFNClassifier

t0 = time.time()
X, Xt = frame(lab.index, "train"), frame(sub["id"], "test")
Xt = Xt[X.columns]
print(f"{X.shape[1]} columns, load {time.time() - t0:.0f}s", flush=True)
for k in map(int, sys.argv[1:]):
    tr, va = folds[k]
    med = X.iloc[tr].median()
    A, V, T = (D.fillna(med).to_numpy(np.float32) for D in (X.iloc[tr], X.iloc[va], Xt))
    clf = TabPFNClassifier(device="cuda", n_estimators=8, ignore_pretraining_limits=True, random_state=0)
    clf.fit(A, y[tr])
    pv = clf.predict_proba(V)[:, 1]
    pt = clf.predict_proba(T)[:, 1]
    np.savez(f"outputs/factory/tabpfn_f{k}.npz", va=pv, test=pt)
    print(f"fold {k}: AUC {roc_auc_score(y[va], pv):.4f}  ({time.time() - t0:.0f}s)", flush=True)
