"""Paired bootstrap of two confirm.py runs (OOF AUC), e.g. combo10 vs combo10g.

    python probe/cmp_confirm.py combo10 combo10g
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

a, b = sys.argv[1:3]
y = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"].to_numpy()
oa, ob = np.load(f"outputs/factory/confirm_{a}.npz")["oof"], np.load(f"outputs/factory/confirm_{b}.npz")["oof"]
rng = np.random.default_rng(0)
d = [roc_auc_score(y[i], ob[i]) - roc_auc_score(y[i], oa[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
print(f"{a} OOF {roc_auc_score(y, oa):.4f}   {b} OOF {roc_auc_score(y, ob):.4f}   gain {roc_auc_score(y, ob) - roc_auc_score(y, oa):+.4f}  "
      f"95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
