"""Which statistic gives the break generator away most? Real break series vs menu "L tuned": C2ST AUC with each
feature alone and with each feature removed.

    python probe/c2st_ablate.py [MENU]
"""
import sys
import time

sys.argv = [sys.argv[0], "--menu", sys.argv[1] if len(sys.argv) > 1 else "L tuned", "--only-break", "1"]
import importlib.util
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

s = importlib.util.spec_from_file_location("bc", "probe/break_c2st.py")
bc = importlib.util.module_from_spec(s); s.loader.exec_module(bc)


def auc(X, y):
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=30)
    return roc_auc_score(y, cross_val_predict(m, X, y, cv=5, method="predict_proba")[:, 1])


t0 = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = set(np.random.default_rng(7).choice(lab.index[lab == 1], 1500, False))
real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
fake = Parallel(n_jobs=30)(delayed(bc.cb.gen)(bc.MENU, 5 * 10 ** 6 + 10 ** 5 + i, True, True) for i in range(1500))
F = pd.DataFrame(Parallel(n_jobs=30)(delayed(bc.feats)(v, n0) for v, n0 in real + fake))
keep = F.cls.isin(["WN", "AR1", "AR2"]).to_numpy()
y = np.r_[np.zeros(len(real)), np.ones(len(fake))][keep]
X = F.loc[keep].drop(columns="cls")
X.columns = [c.replace("(", "").replace(")", "").replace(">", "gt").replace(":", "") for c in X.columns]
full = auc(X, y)
rows = [(c, auc(X[[c]], y), full - auc(X.drop(columns=c), y)) for c in X.columns]
R = pd.DataFrame(rows, columns=["feature", "alone", "drop_gain"]).sort_values("alone", ascending=False)
print(f"all features AUC {full:.3f}")
print(R.to_string(index=False, float_format="%.3f"))
print(f"total {time.time() - t0:.1f}s")
