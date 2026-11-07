"""TabPFN smoke test, fold 0 only (same StratifiedKFold(5, shuffle, 42) as the main runs).
Features = top-K combo10gnj columns by LightGBM gain on the fold-0 train part; context = M random fold-0 train rows.
Reports TabPFN fold-0 AUC, rank corr with the combo10gnj LGB OOF, AUC of the equal rank blend vs LGB alone (paired
bootstrap), and CPU time (2 threads) to predict 1000 rows.

    ~/venvs/tabpfn/bin/python probe/tabpfn_smoke.py [K] [M]
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

import os
MP = os.environ.get("TABPFN_MP", "auto")   # "auto" = default (3.5); or a local .ckpt path such as the v2 weights
TAG = os.environ.get("TABPFN_TAG", "")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
M = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
sys.argv = sys.argv[:1]
t = time.time()
src = open("probe/param_quick.py").read()
exec(src[src.index("lab = pd.read_csv"):src.index("tr, va = next")])      # X (combo10g), y, ld
X = pd.concat([X, ld("outputs/factory/gnoise.parquet"), ld("outputs/factory/gjlr.parquet")], axis=1)
X = X.loc[:, ~X.columns.duplicated() & ~X.columns.str.startswith("__")]
tr, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
base = np.load("outputs/factory/confirm_combo10gnj.npz")["oof"]
print(f"{X.shape[1]} cols, load {time.time() - t:.0f}s", flush=True)
m = lgb.LGBMClassifier(learning_rate=.1, n_estimators=300, num_leaves=31, colsample_bytree=.5, verbose=-1, n_jobs=30).fit(X.iloc[tr], y[tr])
top = pd.Series(m.booster_.feature_importance("gain"), X.columns).sort_values(ascending=False).index[:K]
Xt = X[top].replace([np.inf, -np.inf], np.nan)
med = Xt.iloc[tr].median()
Xt = Xt.fillna(med).to_numpy(np.float32)
ctx = np.random.default_rng(0).choice(tr, min(M, len(tr)), replace=False)
t = time.time()
clf = TabPFNClassifier(device="cuda", model_path=MP, ignore_pretraining_limits=True, random_state=0)
clf.fit(Xt[ctx], y[ctx])
p = clf.predict_proba(Xt[va])[:, 1]
print(f"TabPFN K={K} M={len(ctx)}: fold-0 AUC {roc_auc_score(y[va], p):.4f} (GPU {time.time() - t:.0f}s)  LGB OOF fold-0 {roc_auc_score(y[va], base[va]):.4f}", flush=True)
rb, rp = rankdata(base[va]) / len(va), rankdata(p) / len(va)
bl = (rb + rp) / 2
rng = np.random.default_rng(0)
yy = y[va]
d = [roc_auc_score(yy[i], bl[i]) - roc_auc_score(yy[i], rb[i]) for i in (rng.integers(0, len(va), len(va)) for _ in range(300))]
print(f"rank corr {np.corrcoef(rb, rp)[0, 1]:.3f}  equal blend {roc_auc_score(yy, bl):.4f}  gain {roc_auc_score(yy, bl) - roc_auc_score(yy, rb):+.4f}  "
      f"CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]", flush=True)
if os.environ.get("TABPFN_NOCPU"):
    sys.exit()
torch.set_num_threads(2)
t = time.time()
c2 = TabPFNClassifier(device="cpu", model_path=MP, ignore_pretraining_limits=True, random_state=0)
c2.fit(Xt[ctx], y[ctx])
c2.predict_proba(Xt[va[:200]])
dt = time.time() - t
print(f"CPU 2 threads: 200 rows in {dt:.0f}s -> 4000 test rows ~{dt * 20 / 60:.0f} min (one context)")
json.dump(list(top), open(f"outputs/factory/tabpfn_top{K}{TAG}.json", "w"))
