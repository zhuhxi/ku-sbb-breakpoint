"""Cross-validated blend weights: weights picked on 4 folds (grid), scored on the 5th. Arms: saf(ps), bag6, bag6w, inter1, inter2, tabpfn?"""
import numpy as np, pandas as pd, itertools, os
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy(); r = lambda a: rankdata(a) / len(a)
Z = lambda t: r(np.load(f"outputs/factory/confirm_{t}.npz")["oof"]); I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
arms = {"pairavg": r(Z("pair60") + Z("pair40") + Z("pair80") + Z("pair60saf") + Z("pairb_50_1") + Z("pairb_70_2") + Z("pairb_60_3")), "pair60": Z("pair60"), "ps_saf": Z("ps_saf"), "bag6w": Z("combo10gnjbag6w"), "i1": r(I[1]), "i2": r(I[2])}
arms["tabpfn"] = r(np.load("outputs/factory/tabpfn_oof.npz")["oof"])
names = list(arms); A = np.column_stack([arms[n] for n in names]); print(names)
base = r(A[:, [names.index(n) for n in ("ps_saf", "bag6w", "i2")]] @ np.array([1.5, 1.5, 2]))   # blendB
grid = [w for w in itertools.product([0, 1, 2], repeat=len(names)) if sum(w) > 0]
cv = StratifiedKFold(5, shuffle=True, random_state=7); out = np.zeros(len(y)); picks = []
for tr, va in cv.split(A, y):
    best = max(grid, key=lambda w: roc_auc_score(y[tr], A[tr] @ np.array(w)))
    picks.append(best); out[va] = r(A[va] @ np.array(best))   # rank within fold
    print(best, f"val {roc_auc_score(y[va], A[va] @ np.array(best)):.4f} vs base {roc_auc_score(y[va], base[va]):.4f}", flush=True)
fold_auc = lambda s: np.mean([roc_auc_score(y[va], s[va]) for _, va in cv.split(A, y)])
print(f"mean fold AUC: cv-picked {fold_auc(A @ np.zeros(len(names)) + out):.4f}  base {fold_auc(base):.4f}")
