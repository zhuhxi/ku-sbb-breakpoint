"""Closing-the-loop check of the calibrated no-break generator (menu sampled from scratch, not per-series copies).

    PYTHONPATH=. python probe/menu_check.py --n 1500
Both real label-0 series and generated series are standardized per series (shape only, no offset/scale).
"""
import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from features import build_features

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--jobs", type=int, default=30)
ap.add_argument("--garch", type=int, default=1)
ap.add_argument("--menu", default="")                 # e.g. F: use a calibrated menu from probe/calib_params.py
ap.add_argument("--trend", type=float, default=0.0)   # share of series that get a straight line added
args = ap.parse_args()
MENUS = {"F": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.7, .7), ar2=((-.9, .9), (-.5, .5)),
                   ll=(.02, .1), tp=.08, tr=(.5, 2.5))}   # calibrated in probe/calib_params.py (menu J there)
STRUCT = {"WN": 0.42, "SIN": 0.16, "RW": 0.07, "AR1+obs": 0.15, "AR1": 0.05, "AR2": 0.06, "ARMA11": 0.05, "MA1": 0.04}


def noise(r, n):
    u = r.random()
    if u < 0.05:                                            # bimodal: +-a + N(0, 1)
        return r.choice([-1, 1], n) * r.uniform(1.0, 2.5) + r.standard_normal(n)
    if u < 0.525:
        df = r.uniform(5, 10)
        z = r.standard_t(df, n) / np.sqrt(df / (df - 2))
    else:
        z = r.standard_normal(n)
    if args.garch and r.random() < 0.25:                    # GARCH(1,1), calibrated: a ~ U[.05,.15], b ~ U[.5,.8]
        a, b = r.uniform(0.05, 0.15), r.uniform(0.5, 0.8)
        e, s2 = np.empty(n), 1.0
        for t in range(n):
            if t:
                s2 = (1 - a - b) + a * e[t - 1] ** 2 + b * s2
            e[t] = np.sqrt(s2) * z[t]
        return e
    return z


def gen_menu(seed):
    """Menu from probe/calib_params.py MENUS (structures WN/SIN/RW/AR1/AR2/LL + trend layer)."""
    M = MENUS[args.menu]
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    e, t = noise(r, n), np.arange(n)
    k = r.choice(list(M["p"]), p=np.array(list(M["p"].values())) / sum(M["p"].values()))
    if k == "WN":
        x = e
    elif k == "SIN":
        x = r.uniform(0.5, 3) * np.sin(2 * np.pi * t / r.integers(12, 33) + r.uniform(0, 2 * np.pi)) + e
    elif k == "RW":
        x = np.cumsum(e)
    elif k == "AR1":
        x = lfilter([1], [1, -r.uniform(*M["ar1"])], e)
    elif k == "AR2":
        while True:
            p1, p2 = r.uniform(*M["ar2"][0]), r.uniform(*M["ar2"][1])
            if p2 + p1 < 1 and p2 - p1 < 1:
                break
        x = lfilter([1], [1, -p1, -p2], e)
    else:
        x = np.cumsum(noise(r, n) * r.uniform(*M["ll"])) + e
    if k in ("WN", "AR1", "AR2") and r.random() < M["tp"]:
        x = x + r.choice([-1, 1]) * r.uniform(*M["tr"]) * x.std() * t / n
    n0 = int(r.uniform(0.3, 0.7) * n)
    return pd.DataFrame({"id": f"g{seed}_{k}", "time": t, "value": x, "period": (t >= n0).astype(np.int8)})


def gen(seed):
    if args.menu:
        return gen_menu(seed)
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    kind = r.choice(list(STRUCT), p=np.array(list(STRUCT.values())) / sum(STRUCT.values()))
    e, t = noise(r, n), np.arange(n)
    ar = lambda phi, e: lfilter([1], np.r_[1, -np.asarray(phi)], e)
    if kind == "WN":
        x = e
    elif kind == "SIN":
        x = r.uniform(0.5, 3) * np.sin(2 * np.pi * t / r.integers(12, 33) + r.uniform(0, 2 * np.pi)) + e
    elif kind == "RW":
        x = np.cumsum(e)
    elif kind == "AR1":
        x = ar([r.uniform(-0.85, 0.85)], e)
    elif kind == "AR1+obs":
        x = ar([r.uniform(-0.85, 0.85)], e) + noise(r, n) * r.uniform(0.3, 1.0)
    elif kind == "AR2":
        while True:
            p1, p2 = r.uniform(-1.5, 1.5), r.uniform(-0.8, 0.8)
            if abs(p2) < 1 and p2 + p1 < 1 and p2 - p1 < 1:
                break
        x = ar([p1, p2], e)
    elif kind == "ARMA11":
        x = lfilter([1, r.uniform(-0.9, 0.9)], [1, -r.uniform(-0.85, 0.85)], e)
    else:
        x = lfilter([1, r.uniform(-0.9, 0.9)], [1], e)
    if r.random() < args.trend:                             # straight line, total rise U[0.5, 2.5] x sd of the series
        x = x + r.choice([-1, 1]) * r.uniform(0.5, 2.5) * x.std() * t / n
        kind = kind + "+T"
    n0 = int(r.uniform(0.3, 0.7) * n)
    return pd.DataFrame({"id": f"g{seed}_{kind}", "time": t, "value": x, "period": (t >= n0).astype(np.int8)})


def standardize(df):
    g = df.groupby("id")["value"]
    return df.assign(value=(df["value"] - g.transform("mean")) / g.transform("std"))


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    ids = np.random.default_rng(0).choice(lab.index[lab == 0], args.n, replace=False)
    S = pd.read_parquet("data/train_series.parquet")
    G = pd.concat(Parallel(n_jobs=args.jobs)(delayed(gen)(s) for s in range(args.n)))
    F = build_features(standardize(pd.concat([S[S.id.isin(set(ids))], G])), n_jobs=args.jobs)
    yy = np.array([i.startswith("g") and "_" in i for i in F.index], int)
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=args.jobs)
    pr = cross_val_predict(m, F, yy, cv=5, method="predict_proba")[:, 1]
    imp = pd.Series(m.fit(F, yy).booster_.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    print(f"real vs menu-generated no-break series: AUC {roc_auc_score(yy, pr):.3f}  (0.5 = indistinguishable)")
    print("  tell-tale features:", list(imp.index[:10]))
    kinds = pd.Series([i.split("_", 1)[1].replace("+T", "") if yy[j] else "real" for j, i in enumerate(F.index)])
    print("  how 'fake-looking' each generated kind is (mean predicted fake prob, real = %.2f):" % pr[yy == 0].mean())
    print("  ", {k: round(float(pr[(kinds == k).to_numpy()].mean()), 2) for k in (MENUS[args.menu]["p"] if args.menu else STRUCT)})
    print(f"total {time.time() - t0:.1f}s")
