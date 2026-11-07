"""ABC model choice by classification (Pudlo et al. 2016 style) for the no-break structure menu.

Simulate series from each candidate structure (with the current noise layer: 50% normal / 50% t, df ~ U[5,10]),
compute the 242 features.py statistics on standardized series, train a classifier "which structure made this
series", check on held-out simulations how well each structure is recognised, then apply it to real label-0
series and turn the predicted probabilities into proportions (averaged probabilities, then confusion-adjusted).

    PYTHONPATH=. python probe/abc_rf.py --per 300 --real 3000
"""
import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from scipy.signal import lfilter
from sklearn.model_selection import cross_val_predict

from features import build_features

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--per", type=int, default=300)
ap.add_argument("--real", type=int, default=3000)
ap.add_argument("--jobs", type=int, default=30)
args = ap.parse_args()
CLASSES = ["WN", "AR1", "AR2", "MA1", "ARMA11", "AR1+obs", "RW", "SIN+WN"]


def noise(r, n):
    return r.standard_t(r.uniform(5, 10), n) if r.random() < 0.5 else r.standard_normal(n)


def gen(kind, seed):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    e = noise(r, n)
    t = np.arange(n)
    ar = lambda phi, e: lfilter([1], np.r_[1, -np.asarray(phi)], e)
    if kind == "WN":
        x = e
    elif kind == "AR1":
        x = ar([r.uniform(-0.85, 0.85)], e)
    elif kind == "AR2":
        while True:
            p1, p2 = r.uniform(-1.5, 1.5), r.uniform(-0.8, 0.8)
            if abs(p2) < 1 and p2 + p1 < 1 and p2 - p1 < 1:
                break
        x = ar([p1, p2], e)
    elif kind == "MA1":
        x = lfilter([1, r.uniform(-0.9, 0.9)], [1], e)
    elif kind == "ARMA11":
        x = lfilter([1, r.uniform(-0.9, 0.9)], [1, -r.uniform(-0.85, 0.85)], e)
    elif kind == "AR1+obs":
        x = ar([r.uniform(-0.85, 0.85)], e) + noise(r, n) * r.uniform(0.3, 1.0)
    elif kind == "RW":
        x = np.cumsum(e)
    else:
        x = r.uniform(0.5, 2.0) * np.sin(2 * np.pi * t / r.uniform(12, 35) + r.uniform(0, 6.3)) + e
    n0 = int(r.uniform(0.3, 0.7) * n)
    return pd.DataFrame({"id": f"{kind}_{seed}", "time": t, "value": x, "period": (t >= n0).astype(np.int8)})


def standardize(df):
    g = df.groupby("id")["value"]
    return df.assign(value=(df["value"] - g.transform("mean")) / g.transform("std"))


if __name__ == "__main__":
    t0 = time.time()
    sims = Parallel(n_jobs=args.jobs)(delayed(gen)(k, 10000 * i + s) for i, k in enumerate(CLASSES) for s in range(args.per))
    Fs = build_features(standardize(pd.concat(sims)), n_jobs=args.jobs)
    ys = np.array([CLASSES.index(i.rsplit("_", 1)[0]) for i in Fs.index])
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    ids = np.random.default_rng(0).choice(lab.index[lab == 0], args.real, replace=False)
    S = pd.read_parquet("data/train_series.parquet")
    Fr = build_features(standardize(S[S.id.isin(set(ids))]), n_jobs=args.jobs)[Fs.columns]
    print(f"features: {len(Fs)} simulated + {len(Fr)} real in {time.time() - t0:.1f}s")

    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.5, verbose=-1, n_jobs=args.jobs)
    ps = cross_val_predict(m, Fs, ys, cv=5, method="predict_proba")
    C = np.array([ps[ys == i].mean(0) for i in range(len(CLASSES))])      # soft confusion: true -> predicted
    pd.set_option("display.width", 200)
    print("held-out simulations, average predicted probability (%): true structure (rows) -> predicted (cols)")
    print((100 * pd.DataFrame(C, index=CLASSES, columns=CLASSES)).round(0).astype(int).to_string())
    pr = m.fit(Fs, ys).predict_proba(Fr)
    raw = pr.mean(0)
    adj, res = nnls(C.T, raw)
    print("\nreal no-break series, structure share (%):")
    print(pd.DataFrame({"averaged prob": 100 * raw, "confusion-adjusted": 100 * adj / adj.sum()}, index=CLASSES).round(1).to_string())
    print(f"adjustment residual {res:.3f}; total {time.time() - t0:.1f}s")
