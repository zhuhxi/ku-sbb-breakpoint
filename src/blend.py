"""Rank-blend LightGBM (train.py) and CNN (cnn.py) predictions; weight chosen on OOF AUC."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

out = Path("outputs")
oof = pd.read_csv(out / "oof.csv", dtype={"id": str})
y = oof["label"].to_numpy()
o_lgb, o_cnn = oof["oof_lgb"].to_numpy(), np.load(out / "oof_cnn.npy")
sub = pd.read_csv(out / "submission.csv", dtype={"id": str})
p_lgb, p_cnn = sub["break_probability"].to_numpy(), np.load(out / "pred_cnn.npy")

r = lambda v: rankdata(v) / len(v)
grid = np.linspace(0, 1, 21)
scores = [roc_auc_score(y, (1 - w) * r(o_lgb) + w * r(o_cnn)) for w in grid]
w = grid[int(np.argmax(scores))]
print(f"lgb {roc_auc_score(y, o_lgb):.4f}  cnn {roc_auc_score(y, o_cnn):.4f}  blend w_cnn={w:.2f} {max(scores):.4f}")

sub["break_probability"] = (1 - w) * r(p_lgb) + w * r(p_cnn)
assert np.isfinite(sub["break_probability"]).all() and sub["break_probability"].between(0, 1).all()
sub.to_csv(out / "submission_blend.csv", index=False)
print("wrote", out / "submission_blend.csv")
