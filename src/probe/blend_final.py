"""CV-picked weights (grid {0,1,2,3}) over: pairavg (all pair bags), new inter arms (wt162: all91k, top800), old inter arms, ps_saf, bag6w.
Weights picked on 4 folds (StratifiedKFold seed 7), scored on the 5th; then fixed weights = most common pick; writes submission."""
import numpy as np, pandas as pd, glob, itertools, sys
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy(); sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str}); r = lambda a: rankdata(a) / len(a)
fs = sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")); Z = lambda t, p: r(np.load(f"outputs/factory/confirm_{t}.npz")[p])
IO = {"oof": np.load("outputs/brute/oof_lgb_inter_gnjt162.npy"), "test": np.load("outputs/brute/test_lgb_inter_gnjt162.npy")}
IN = {"oof": np.load("outputs/brute/oof_lgb_inter_gnjwt162.npy"), "test": np.load("outputs/brute/test_lgb_inter_gnjwt162.npy")}
def arms(p): return {"pairavg": r(sum(r(np.load(f)[p]) for f in fs)), "new_all": r(IN[p][1]), "new_top": r(IN[p][2]), "old_all": r(IO[p][1]), "old_top": r(IO[p][2]), "ps_saf": Z("ps_saf", p)}
A = arms("oof"); names = list(A); M = np.column_stack([A[n] for n in names]); print(names, {n: round(roc_auc_score(y, A[n]), 4) for n in names})
base = r(2 * A["pairavg"] + A["old_top"])
grid = [w for w in itertools.product([0, 1, 2, 3], repeat=len(names)) if sum(w) > 0]
cv = list(StratifiedKFold(5, shuffle=True, random_state=7).split(M, y)); picks = []; out = []
for tr, va in cv:
    best = max(grid, key=lambda w: roc_auc_score(y[tr], M[tr] @ np.array(w))); picks.append(best)
    out.append((roc_auc_score(y[va], M[va] @ np.array(best)), roc_auc_score(y[va], base[va]))); print(best, f"val {out[-1][0]:.4f} vs blendD {out[-1][1]:.4f}")
print(f"cv-picked mean {np.mean([o[0] for o in out]):.4f} vs blendD {np.mean([o[1] for o in out]):.4f}")
w = np.mean(picks, 0); s = r(M @ w); rng = np.random.default_rng(0)
d = [roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], base[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
print("avg weights", dict(zip(names, w.round(2))), f"OOF {roc_auc_score(y, s):.4f} vs blendD {roc_auc_score(y, base):.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]")
T = arms("test"); t = r(np.column_stack([T[n] for n in names]) @ w)
pd.DataFrame({"id": sub.id, "break_probability": t}).to_csv("outputs/submission_blendE.csv", index=False); print("wrote outputs/submission_blendE.csv")
