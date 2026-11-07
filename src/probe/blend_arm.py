"""Add a NEW arm to the current best blend (saf + bag6 + inter 1,2; OOF .8582, public .85830).
    python probe/blend_arm.py path/to/oof.npy [more oof.npy ...]
Each file: OOF probabilities for the 24k train rows in data/train_labels.csv order. Fixed weights, paired bootstrap 500."""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy()
r = lambda a: rankdata(a) / len(a)
O = lambda t: r(np.load(f"outputs/factory/confirm_{t}.npz")["oof"])
I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
best = r((1.5 * O("ps_saf") + 1.5 * O("combo10gnjbag6w") + 2 * r(I[2])) / 5)   # blendB, public .85833
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(500)]
def ci(a, b):
    d = np.array([roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in B]); return np.percentile(d, [2.5, 97.5])
print(f"best {roc_auc_score(y, best):.4f}")
for f in sys.argv[1:]:
    a = np.load(f); a = a if a.ndim == 1 else a.ravel()
    ra = r(a); print(f"\n{f}: alone {roc_auc_score(y, a):.4f}  rank corr {np.corrcoef(ra, best)[0,1]:.3f}")
    for w in (.1, .2, .3):
        s = (1 - w) * best + w * ra; g = roc_auc_score(y, s) - roc_auc_score(y, best); lo, hi = ci(s, best)
        print(f"  w {w:.1f}: {g:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
