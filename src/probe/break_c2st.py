"""Real-vs-fake check for the break menu (C2ST). Before-vs-after statistics (probe/hidden_diag.py) for real and
generated series whose pre-boundary part is WN / AR1 / AR2; LightGBM 5-fold AUC, 0.5 = indistinguishable.
Reference: the same test on no-break series (real label 0 vs the calibrated no-break generator).

    python probe/break_c2st.py --n 1500
"""
import argparse
import importlib.util
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--level-unit", default="series")   # series: shift in sd of the series; noise: in sd of the innovations
ap.add_argument("--only-break", type=int, default=0)
ap.add_argument("--menu", default="")             # name in calib_break.BREAKS instead of the tuned menu
ap.add_argument("--strong-level", type=float, default=0)   # >0: replace the strong level sizes 1.8-2.2 by this value
args = ap.parse_args()
sys.argv = [sys.argv[0], "--n", "1"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
hd = load("hd", "probe/hidden_diag.py")
tdp = load("tdp", "probe/tune_dep.py")
MENU = dict(tdp.menu((.1, .25), .5, .1))            # tuned: spread weak x1.1-1.35 / strong x2 (1/4); dep weak .1-.25 / strong .5 (1/10)


def feats(v, n0):
    f = hd.one(v, n0)
    f["n"], f["frac"] = len(v), n0 / len(v)
    return f


def c2st(real, fake):
    F = pd.DataFrame(Parallel(n_jobs=30)(delayed(feats)(v, n0) for v, n0 in real + fake))
    keep = F.cls.isin(["WN", "AR1", "AR2"]).to_numpy()
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))][keep]
    X = F.loc[keep].drop(columns="cls")
    X.columns = [c.replace("(", "").replace(")", "").replace(">", "gt").replace(":", "") for c in X.columns]
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=30)
    p = cross_val_predict(m, X, y, cv=5, method="predict_proba")[:, 1]
    imp = pd.Series(m.fit(X, y).booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    return roc_auc_score(y, p), list(imp.index[:4])


MENU["level_unit"] = args.level_unit
if args.menu:
    MENU = dict(cb.BREAKS[args.menu])
if args.strong_level:
    MENU["level"] = ([v if v < 1 else round(args.strong_level + (v - 2.0) / 2, 2) for v in MENU["level"][0]],)

if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    r = np.random.default_rng(7)
    S = pd.read_parquet("data/train_series.parquet")
    out = {}
    for y in ((1,) if args.only_break else (0, 1)):
        ids = set(r.choice(lab.index[lab == y], args.n, False))
        real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
        fake = Parallel(n_jobs=30)(delayed(cb.gen)(MENU, 5 * 10 ** 6 + 10 ** 5 * y + s, bool(y), True) for s in range(args.n))
        out[y] = c2st(real, fake)
    if 0 in out:
        print(f"no-break (reference): real vs generated AUC {out[0][0]:.3f}   tell-tale {out[0][1]}")
    print(f"break    (menu, level unit {args.level_unit}):      real vs generated AUC {out[1][0]:.3f}   tell-tale {out[1][1]}")
    print(f"total {time.time() - t0:.1f}s")
