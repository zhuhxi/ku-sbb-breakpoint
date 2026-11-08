"""Blend check of an extra arm group (several weights) against blendN (.8701)."""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
exec(open("probe/blend_l.py").read().split("ex = r(")[0])
NU = ["tpf60sidemnr_0","tpf70sidemnr_2","tpf50sidemnr_1","tpf65sidemnr_3","tpf40sidemnr_5","tpf80sidemnr_4","tpf60sidemn_0","tpf70sidemn_2","tpf60sidemn2_0"]
CB = ["tpf60sidemnrcb_0","tpf70sidemnrcb_2","tpf50sidemnrcb_1","tpf65sidemnrcb_3","tpf40sidemnrcb_5","tpf80sidemnrcb_4","tpf55sidemnrcb_6","tpf75sidemnrcb_7"]
g = lambda T: r(sum(r(np.load(f"outputs/arms/{t}_oof.npy")) for t in T)); nu = g(NU); cb = g(CB); BN = r(core + 6 * sw + 12 * mf + 24 * nu + 144 * cb)
ex = r(sum(r(np.load(f)) for f in sys.argv[1:])); print("blendN", round(roc_auc_score(y, BN), 4), "| extra", round(roc_auc_score(y, ex), 4), "corr", round(np.corrcoef(ex, BN)[0, 1], 4))
for w in (12, 24, 48, 96):
    s = r(core + 6 * sw + 12 * mf + 24 * nu + 144 * cb + w * ex); d = [roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], BN[i]) for i in B]
    print(f"w {w:3d}: {roc_auc_score(y, s):.4f} CI [{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}] folds {sum(roc_auc_score(y[va], s[va]) > roc_auc_score(y[va], BN[va]) for _, va in cv)}/5")
