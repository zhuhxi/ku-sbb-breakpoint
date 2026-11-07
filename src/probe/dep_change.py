"""Dependence breaks on series that are white-noise-like BEFORE the boundary (|pre lag-1 corr| < .1):
post lag-1 corr, label 1 minus label 0 (excess), real vs a generator menu. Also: share of label-1 series with
no visible change (|level| < .2, |log spread| < .15, |post ac1| < .1).

    python probe/dep_change.py MENU
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()
menu = sys.argv[1]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
sys.argv = sys.argv[:1]
cb = load("cb", "probe/calib_break.py")
ac = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]


def m(v, n0, lab):
    a, b = v[:n0], v[n0:]
    if abs(ac(a)) < .1:
        return dict(dep=ac(b), level=abs(b.mean() - a.mean()) / a.std(), spread=abs(np.log(b.std() / a.std())), label=lab)


S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id")["label"]
R = pd.DataFrame([r for i, g in S.groupby("id", sort=False) if (r := m(g.value.to_numpy(), int((g.period == 0).sum()), y[i]))])
B = dict(cb.BREAKS[menu])
F = pd.DataFrame([r for r in Parallel(n_jobs=-1, batch_size=64)(delayed(lambda s: m(*cb.gen(B, 2 * 10 ** 6 + s, bool(s % 2), True), s % 2))(s) for s in range(24000)) if r])
edges = np.r_[-1, -.5, -.3, -.2, -.1, .1, .2, .3, .5, 1]


def excess(D, scale):
    k1 = (D.label == 1).sum()
    return (np.histogram(D.dep[D.label == 1], edges)[0] - np.histogram(D.dep[D.label == 0], edges)[0] * k1 / (D.label == 0).sum()) * scale / k1


k1r = (R.label == 1).sum()
er, ef = excess(R, k1r), excess(F, k1r)
print(f"{menu}: pre-white-noise-like  real {len(R)} (label1 {k1r})  fake {len(F)}")
print("post ac1 bin     fake excess   real excess")
for k in range(len(edges) - 1):
    print(f"  {edges[k]:+.1f}..{edges[k+1]:+.1f}   {ef[k]:+7.0f}     {er[k]:+7.0f}")
inv = lambda D: ((D.level < .2) & (D.spread < .15) & (D.dep.abs() < .1))
for name, D in (("real", R), ("fake", F)):
    print(f"{name}: no visible change  label1 {inv(D)[D.label == 1].mean():.3f}  label0 {inv(D)[D.label == 0].mean():.3f}"
          f"  -> label1 with nothing beyond label0 rate ~ {(inv(D)[D.label == 1].mean() / inv(D)[D.label == 0].mean()):.2f} of label1")
print(f"{time.time() - t0:.1f}s")
