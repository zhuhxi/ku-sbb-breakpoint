"""Rank blends of the combo10gnj LightGBM with the XGBoost / CatBoost runs of stack_models.py (same folds).
Main claim = fixed equal weights (no weights fitted on the OOF); the weight grid is only for orientation.
Paired bootstrap (500) of the gain over the LightGBM alone.

    python blend_stack.py [--write NAME]   # --write: save outputs/submission_NAME.csv with the equal-weight 3-model blend
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

y = pd.read_csv("data/train_labels.csv")["label"].to_numpy()
M = {"lgb": np.load("outputs/factory/confirm_combo10gnj.npz")}
for k in ("xgb", "cat"):
    try:
        M[k] = np.load(f"outputs/factory/stack_{k}.npz")
    except FileNotFoundError:
        pass
R = {k: (rankdata(v["oof"]) / len(y), rankdata(v["test"]) / len(v["test"])) for k, v in M.items()}
base = R["lgb"][0]
rng = np.random.default_rng(0)
B = [rng.integers(0, len(y), len(y)) for _ in range(500)]


def report(name, o):
    d = [roc_auc_score(y[i], o[i]) - roc_auc_score(y[i], base[i]) for i in B]
    print(f"{name:28s} OOF {roc_auc_score(y, o):.4f}  gain {roc_auc_score(y, o) - roc_auc_score(y, base):+.4f}  "
          f"CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")


for k in R:
    print(f"{k:28s} OOF {roc_auc_score(y, R[k][0]):.4f}   corr with lgb {np.corrcoef(R[k][0], base)[0, 1]:.3f}")
for k in [k for k in R if k != "lgb"]:
    report(f"lgb + {k} (equal)", (base + R[k][0]) / 2)
if len(R) == 3:
    report("lgb + xgb + cat (equal)", sum(r[0] for r in R.values()) / 3)
    print("orientation only (weights chosen on the OOF are optimistic):")
    for w in (.5, .6, .7):
        report(f"  lgb {w:.1f}, others {(1 - w) / 2:.2f} each", w * base + (1 - w) / 2 * (R["xgb"][0] + R["cat"][0]))
if "--write" in sys.argv and len(R) == 3:
    name = sys.argv[sys.argv.index("--write") + 1]
    sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
    sub["break_probability"] = sum(r[1] for r in R.values()) / 3
    assert list(sub.columns) == ["id", "break_probability"] and sub.break_probability.between(0, 1).all()
    sub.to_csv(f"outputs/submission_{name}.csv", index=False)
    print("wrote", f"outputs/submission_{name}.csv")
