"""Break shape: align series at the boundary, standardize by the pre part, flip so the jump is upward, average.
Top 10% by |relative jump|: real label 1, fake label 1 (shell menu), real label 0 (selection control).
Random-walk-like series (pre lag-1 corr > .9) are shown apart; the other curves exclude them.

    python probe/break_profile.py   # -> outputs/probe/break_profile.png
"""
import importlib.util
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

t0 = time.time()
W = 100
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")


def aligned(v, n0):
    a = v[:n0]
    z = (v - a.mean()) / a.std()
    j = z[n0:].mean()
    z = z * np.sign(j)
    out = np.full(2 * W, np.nan)
    lo, hi = max(0, n0 - W), min(len(v), n0 + W)
    out[lo - n0 + W:hi - n0 + W] = z[lo:hi]
    za = (a - a.mean()) / a.std()
    return abs(j), out, np.corrcoef(za[1:], za[:-1])[0, 1]


def top(rows, frac=.1, rw=False):
    rows = [r for r in rows if (r[2] > .9) == rw]
    j = np.array([r[0] for r in rows]); P = np.array([r[1] for r in rows])
    return np.nanmean(P[j >= np.quantile(j, 1 - frac)], 0)


S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
real = {0: [], 1: []}
for i, g in S.groupby("id", sort=False):
    real[y[i]].append(aligned(g.value.to_numpy(), int((g.period == 0).sum())))
B = dict(cb.BREAKS["L abc40 ns35 shell"])
fake = {lab: [aligned(*cb.gen(B, 8 * 10 ** 6 + 2 * s + lab, bool(lab), True)) for s in range(3000)] for lab in (0, 1)}
print("random-walk-like share: real", np.mean([r[2] > .9 for r in real[0] + real[1]]).round(3), " fake", np.mean([r[2] > .9 for r in fake[0] + fake[1]]).round(3))
C = {"real break": top(real[1]), "fake break": top(fake[1]), "real no-break": top(real[0]), "fake no-break": top(fake[0]),
     "real RW-like break": top(real[1], rw=True), "real RW-like no-break": top(real[0], rw=True)}
k = np.arange(-W, W)
pick = [-50, -10, -2, -1, 0, 1, 2, 5, 10, 20, 50, 99]
print("offset            ", pick)
for name, c in C.items():
    print(f"{name:24s}", np.round(c[np.array(pick) + W], 2))
plt.figure(figsize=(9, 4))
for name, c in C.items():
    plt.plot(k, c, label=name)
plt.axvline(0, color="k", lw=.5); plt.legend(); plt.xlabel("time from boundary"); plt.ylabel("pre-standardized, jump up")
plt.tight_layout(); plt.savefig("outputs/probe/break_profile.png", dpi=90)
print(f"{time.time() - t0:.1f}s")
