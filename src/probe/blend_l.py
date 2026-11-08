"""Blend check of extra arms (as one new group, several weights) against blendL (.8688)."""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
exec(open("probe/blend_k.py").read().split("rng=np.random")[0])
MF = ("tpf60sidemf1_0", "tpf70sidemf1_2", "tpf50sidemf1_1", "tpf65sidemf1_3", "tpf40sidemf1_5", "tpf80sidemf1_4", "tpf55sidemf1_6")
mf = r(sum(r(np.load(f"outputs/arms/{t}_oof.npy")) for t in MF)); sw = r(sum(r(np.load(f)) for f in w6)); BL = r(core + 24 * sw + 36 * mf)
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(300)]; cv = list(StratifiedKFold(5, shuffle=True, random_state=7).split(y, y))
ex = r(sum(r(np.load(f)) for f in sys.argv[1:])); print("blendL", round(roc_auc_score(y, BL), 4), "| extra group", round(roc_auc_score(y, ex), 4), "corr", round(np.corrcoef(ex, BL)[0, 1], 4))
for w in (6, 12, 24, 36):
    s = r(core + 24 * sw + 36 * mf + w * ex); d = [roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], BL[i]) for i in B]
    print(f"w {w:3d}: {roc_auc_score(y, s):.4f} CI [{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}] folds {sum(roc_auc_score(y[va], s[va]) > roc_auc_score(y[va], BL[va]) for _, va in cv)}/5")
