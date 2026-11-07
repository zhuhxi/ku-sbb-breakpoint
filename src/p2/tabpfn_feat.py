"""2nd-place step: TabPFN on the 8 global stats (X1). For each OUTER fold k: inner KFold(5, shuffle, 42) on the outer-train part
gives OOF values there; the outer-val (and test) value is the mean of the 5 inner models (their train()/infer()).
    python p2/tabpfn_feat.py K ...  -> outputs/p2/tabpfn_fold{K}.npz
"""
import sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from tabpfn import TabPFNClassifier

t0 = time.time()
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy()
X1 = pd.read_parquet("outputs/p2/x1.parquet")
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
Xtr, Xte = X1.loc[L.id].to_numpy(), X1.loc[sub.id].to_numpy()
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, y))
for K in map(int, sys.argv[1:]):
    tr, va = folds[K]
    f_tr = np.zeros(len(tr)); f_va = np.zeros(len(va)); f_te = np.zeros(len(Xte))
    for a, b in KFold(5, shuffle=True, random_state=42).split(tr):
        m = TabPFNClassifier(device="cuda", random_state=0, ignore_pretraining_limits=True).fit(Xtr[tr[a]], y[tr[a]])
        f_tr[b] = m.predict_proba(Xtr[tr[b]])[:, 1]
        f_va += m.predict_proba(Xtr[va])[:, 1] / 5; f_te += m.predict_proba(Xte)[:, 1] / 5
    np.savez(f"outputs/p2/tabpfn_fold{K}.npz", tr=f_tr, va=f_va, te=f_te)
    print(f"fold {K} done {time.time() - t0:.0f}s", flush=True)
