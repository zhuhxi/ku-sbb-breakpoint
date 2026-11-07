"""Current best (saf + bag6 + inter 1,2) vs the same blend with pseudo-label students (ps_saf, ps_bag6). Paired bootstrap 500.
Writes outputs/submission_ps_inter.csv (NOT submitted)."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy(); sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
r = lambda a: rankdata(a) / len(a); Z = lambda t: np.load(f"outputs/factory/confirm_{t}.npz")
I, It = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy"), np.load("outputs/brute/test_lgb_inter_gnjt162.npy")
def bl(tags, part):
    arms = [r(Z(t)[part]) for t in tags] + [r((I if part == "oof" else It)[j]) for j in (1, 2)]; return r(np.mean(arms, 0))
base = bl(["combo10gnjsaf", "combo10gnjbag6"], "oof")
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(500)]
print(f"best {roc_auc_score(y, base):.4f}")
for tags in (["ps_saf_teacher", "ps_bag6_teacher"], ["ps_saf", "combo10gnjbag6"], ["combo10gnjsaf", "ps_bag6"], ["ps_saf", "ps_bag6"],
             ["combo10gnjsaf", "combo10gnjbag6", "ps_saf", "ps_bag6"]):
    try: s = bl(tags, "oof")
    except FileNotFoundError: continue
    d = np.array([roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], base[i]) for i in B])
    print(f"{'+'.join(tags):60s} {roc_auc_score(y, s):.4f}  {roc_auc_score(y, s) - roc_auc_score(y, base):+.4f}  CI [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  folds-up n/a")
try:
    t = bl(["ps_saf", "ps_bag6"], "test"); pd.DataFrame({"id": sub.id, "break_probability": t}).to_csv("outputs/submission_ps_inter.csv", index=False); print("wrote outputs/submission_ps_inter.csv")
except FileNotFoundError: pass
