"""Does TabPFN add to the current best? Current best = equal rank blend of the combo10gnj LGB and interaction arms 1, 2
(outputs/brute/oof_lgb_inter_gnjt162.npy rows 1, 2; public 0.85593). Fixed-weight blends only, paired bootstrap (500).

    python probe/blend_tabpfn.py [--write NAME]   # --write: outputs/submission_NAME.csv with "best + TabPFN, equal 4"
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

y = pd.read_csv("data/train_labels.csv")["label"].to_numpy()
r = lambda v: rankdata(v) / len(v)
L = np.load("outputs/factory/confirm_combo10gnj.npz")
Io, It = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy"), np.load("outputs/brute/test_lgb_inter_gnjt162.npy")
T = np.load("outputs/factory/tabpfn_oof.npz")
oof = {"lgb": r(L["oof"]), "a1": r(Io[1]), "a2": r(Io[2]), "pfn": r(T["oof"])}
tst = {"lgb": r(L["test"]), "a1": r(It[1]), "a2": r(It[2]), "pfn": r(T["test"])}
best = (oof["lgb"] + oof["a1"] + oof["a2"]) / 3
rng = np.random.default_rng(0)
B = [rng.integers(0, len(y), len(y)) for _ in range(500)]


def rep(nm, o, ref=best):
    d = [roc_auc_score(y[i], o[i]) - roc_auc_score(y[i], ref[i]) for i in B]
    print(f"{nm:34s} OOF {roc_auc_score(y, o):.4f}  gain {roc_auc_score(y, o) - roc_auc_score(y, ref):+.4f}  "
          f"CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  folds up {fu(o, ref)}")


def fu(o, ref):
    from sklearn.model_selection import StratifiedKFold
    return sum(roc_auc_score(y[va], o[va]) > roc_auc_score(y[va], ref[va]) for _, va in StratifiedKFold(5, shuffle=True, random_state=42).split(y, y))


print(f"TabPFN OOF {roc_auc_score(y, oof['pfn']):.4f}  LGB {roc_auc_score(y, oof['lgb']):.4f}  current best {roc_auc_score(y, best):.4f}")
print("rank corr with best:", round(np.corrcoef(best, oof["pfn"])[0, 1], 3), " with lgb:", round(np.corrcoef(oof["lgb"], oof["pfn"])[0, 1], 3))
rep("lgb + pfn (equal) vs lgb", (oof["lgb"] + oof["pfn"]) / 2, oof["lgb"])
rep("best + pfn: equal 4 arms", (oof["lgb"] + oof["a1"] + oof["a2"] + oof["pfn"]) / 4)
rep("best .75 + pfn .25 (same as above)", .75 * best + .25 * oof["pfn"])
rep("best .85 + pfn .15", .85 * best + .15 * oof["pfn"])
if "--write" in sys.argv:
    name = sys.argv[sys.argv.index("--write") + 1]
    sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
    sub["break_probability"] = (tst["lgb"] + tst["a1"] + tst["a2"] + tst["pfn"]) / 4
    assert sub.break_probability.between(0, 1).all()
    sub.to_csv(f"outputs/submission_{name}.csv", index=False)
    print("wrote", f"outputs/submission_{name}.csv")
