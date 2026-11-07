"""Are the "no visible change" breaks a separate invisible class (label-noise-like) or the small end of a continuous
size distribution? Best model's OOF score (outputs/factory/confirm_combo10gnj.npz), share of label 1 per score
ventile. Separate invisible class -> the low ventiles sit on a flat plateau (those positives look exactly like
negatives); continuous small breaks -> the share keeps climbing from the very bottom.

    python probe/invisible.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv("data/train_labels.csv")["label"].to_numpy()
o = np.load("outputs/factory/confirm_combo10gnj.npz")["oof"]
b = pd.qcut(pd.Series(o).rank(method="first"), 20, labels=False)
T = pd.DataFrame({"y": y, "bin": b}).groupby("bin").y.mean()
print("OOF AUC", round(roc_auc_score(y, o), 4), " label rate", round(y.mean(), 3))
print("share of label 1 per score ventile (low -> high):")
print(" ".join(f"{v:.2f}" for v in T))
lo = T.iloc[:8]
print("bottom 8 ventiles: mean", round(lo.mean(), 3), " slope per ventile", round(np.polyfit(range(8), lo, 1)[0], 4),
      "  bottom 4 vs ventiles 5-8:", round(T.iloc[:4].mean(), 3), round(T.iloc[4:8].mean(), 3))
