"""Smaller level jumps, or fewer of them? Real stable-group |mean after - mean before| / sd before, label 1 vs 0,
against two fake menus with about the same label AUC: H1 every jump scaled down, H2 full-size jumps on fewer breaks.

    python probe/level_shape.py
"""
import importlib.util
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

t0 = time.time()
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
S = pd.read_parquet("data/train_series.parquet", columns=["id", "value", "period"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
gs = pd.read_parquet("outputs/factory/genA.parquet")["g_struct"]
g = S.groupby(["id", "period"]).value.agg(["mean", "std"]).unstack()
R = pd.DataFrame({"level": (g["mean"][1] - g["mean"][0]).abs() / g["std"][0], "y": y})
R = R[gs.reindex(R.index).isin([0, 1, 2, 3])]
base = cb.BREAKS["L abc40 ns35 shell s6"]
SM, MD, LG = (list(np.round(np.linspace(.01, .49, 76), 3)), list(np.round(np.linspace(.8, .9, 11), 3)),
              list(np.round(np.linspace(1.8, 2.2, 13), 3)))
MENUS = {"H1 smaller (level x.3)": dict(base, shrink=.6, lshrink=.3),
         "H2 fewer (level on 15%)": dict(base, shrink=.6, switches=dict(level=.15, spread=.35, dep=.2)),
         "H3 no small, level 40%": dict(base, shrink=.6, level=(MD + LG,)),
         "H4 no small, level 20%": dict(base, shrink=.6, level=(MD + LG,), switches=dict(level=.2, spread=.35, dep=.2)),
         "H5 small 1/4, level 25%": dict(base, shrink=.6, level=(SM[::4] + MD + LG,), switches=dict(level=.25, spread=.35, dep=.2))}


def one(B, s):
    x, n0, info = cb.gen(B, 7 * 10 ** 6 + s, bool(s % 2), "info")
    if info["structure"] not in ("WN", "AR1", "AR2"):
        return None
    a, b = x[:n0], x[n0:]
    return dict(level=abs(b.mean() - a.mean()) / a.std(), y=s % 2)


def show(name, D):
    q0 = D.level[D.y == 0]
    q1 = D.level[D.y == 1]
    qs = [.25, .5, .75, .9, .95]
    lift = [q1.quantile(q) / q0.quantile(q) for q in qs]
    tail = (q1 > q0.quantile(.95)).mean()
    print(f"{name:26s} AUC {roc_auc_score(D.y, D.level):.3f}  label1/label0 at q25/50/75/90/95 "
          + " ".join(f"{v:.2f}" for v in lift) + f"   label1 above label0 q95: {tail:.1%}")


show("REAL stable", R)
bs = np.random.default_rng(0)
B = np.array([[(lambda D: [D.level[D.y == 1].quantile(q) / D.level[D.y == 0].quantile(q) for q in (.25, .5, .9)])(R.sample(frac=1, replace=True, random_state=bs.integers(1 << 30)))
               for _ in range(50)]])[0]
print(f"{'  real 90% range':26s} q25 {np.quantile(B[:, 0], .05):.2f}-{np.quantile(B[:, 0], .95):.2f}  q50 {np.quantile(B[:, 1], .05):.2f}-{np.quantile(B[:, 1], .95):.2f}"
      f"  q90 {np.quantile(B[:, 2], .05):.2f}-{np.quantile(B[:, 2], .95):.2f}")
for name, B in MENUS.items():
    D = pd.DataFrame([d for d in Parallel(n_jobs=-1, batch_size=32)(delayed(one)(B, s) for s in range(4000)) if d])
    show(name, D)
print(f"{time.time() - t0:.1f}s")
