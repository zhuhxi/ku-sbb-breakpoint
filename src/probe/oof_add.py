"""5-8 s check: does a new feature add information beyond the current best model (combo10gnj LGB OOF, 888 cols)?
Logistic on [logit(base OOF)] vs [logit(base OOF) + new cols (rank-normalised)], 5-fold, on the first N train series.

    python probe/oof_add.py N
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "probe")
from joblib import Parallel, delayed
from kind_lr import one

t0 = time.time()
N = int(sys.argv[1])
L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
base = np.load("outputs/factory/confirm_combo10gnj.npz")["oof"]
ids = L["id"].to_numpy()[:N]
S = pd.read_parquet("data/train_series.parquet")
S = S[S["id"].astype(str).isin(set(ids))].sort_values(["id", "time"])
G = {str(k): (g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)}
F = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(one)(*G[i]) for i in ids)).astype(float)
y = L["label"].to_numpy()[:N]
b = np.clip(base[:N], 1e-6, 1 - 1e-6)
Xb = np.log(b / (1 - b))[:, None]
Xn = np.column_stack([Xb] + [(F[c].rank().to_numpy() / N - .5) for c in F])
o = {"base": np.zeros(N), "+new": np.zeros(N)}
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xb, y):
    for k, X in (("base", Xb), ("+new", Xn)):
        o[k][va] = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], y[tr]).decision_function(X[va])
rng = np.random.default_rng(0)
d = [roc_auc_score(y[i], o["+new"][i]) - roc_auc_score(y[i], o["base"][i]) for i in (rng.integers(0, N, N) for _ in range(300))]
print(f"N {N}  base {roc_auc_score(y, o['base']):.4f}  +new {roc_auc_score(y, o['+new']):.4f}  "
      f"gain {roc_auc_score(y, o['+new']) - roc_auc_score(y, o['base']):+.4f}  CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  {time.time() - t0:.1f}s")
