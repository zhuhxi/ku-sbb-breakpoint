"""10-second stand-in for the full-feature no-break C2ST: real label 0 vs generated no-break, on the 40 simple
whole-series stats of probe/nb_diff.py + the factory fe / fe2 columns (boundary prediction-error family, the strongest
non-bit discriminators in probe/c2st_full.py).

    python probe/c2st_mini.py N MENU [MENU ...]
"""
import importlib.util
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")
N, menus = int(sys.argv[1]), sys.argv[2:]
sys.argv = [sys.argv[0]]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
fa = load("fa", "factory.py")
src = open("probe/nb_diff.py").read()
exec(src[src.index("def acf"):src.index("t0 = time.time()")])          # feats()


def allf(x, n0):
    return {**feats(x, n0), **{f"fe_{k}": v for k, v in fa.fe(x, n0).items()}, **{f"fe2_{k}": v for k, v in fa.fe2(x, n0).items()}}


t0 = time.time()
L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
ids = set(L.loc[L.label == 0, "id"].to_numpy()[:N])
S = pd.read_parquet("data/train_series.parquet")
S = S[S["id"].astype(str).isin(ids)].sort_values(["id", "time"])
real = [(g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for _, g in S.groupby("id", sort=False)]
Fr = pd.DataFrame(Parallel(n_jobs=30, batch_size=16)(delayed(allf)(x, n0) for x, n0 in real)).astype(float)
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, verbose=-1, n_jobs=30)
for m in menus:
    fake = Parallel(n_jobs=30, batch_size=32)(delayed(cb.gen)(cb.BREAKS[m], 10_000 + s, False, True) for s in range(N))
    Ff = pd.DataFrame(Parallel(n_jobs=30, batch_size=16)(delayed(allf)(x, n0) for x, n0 in fake)).astype(float)
    F = pd.concat([Fr, Ff], ignore_index=True).replace([np.inf, -np.inf], np.nan)
    y = np.r_[np.zeros(len(Fr)), np.ones(len(Ff))]
    out = []
    for nm, cols in (("all", list(F.columns)), ("fe/fe2 only", [c for c in F if c.startswith("fe")]),
                     ("near20 only", [c for c in F if "near20" in c])):
        p = cross_val_predict(lgb.LGBMClassifier(**P), F[cols], y, cv=5, method="predict_proba")[:, 1]
        out.append(f"{nm} {roc_auc_score(y, p):.3f}")
    print(f"{m:5s} ({F.shape[1]} cols): " + "   ".join(out) + f"   {time.time() - t0:.1f}s", flush=True)
