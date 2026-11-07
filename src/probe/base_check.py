"""Outer base level / size of real series (pre-boundary part): how drawn, independent?, jump scaled by which size.

    python probe/base_check.py   # -> outputs/probe/base_check.png
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

t0 = time.time()
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
rows = {}
for i, g in S.groupby("id", sort=False):
    v, p = g.value.to_numpy(), g.period.to_numpy()
    a, b = v[p == 0], v[p == 1]
    rows[i] = dict(m0=a.mean(), s0=a.std(), z0=np.diff(a).std() / np.sqrt(2), m1=b.mean(), s1=b.std())
D = pd.DataFrame.from_dict(rows, orient="index").join(y.rename("y"))
D["ls0"] = np.log10(D.s0)
q = [.01, .05, .25, .5, .75, .95, .99]
print("1) pre mean   quantiles", np.round(D.m0.quantile(q).to_numpy(), 2))
print("   pre log10 sd quantiles", np.round(D.ls0.quantile(q).to_numpy(), 2), " sd:", np.round(D.s0.quantile(q).to_numpy(), 2))
print("   mean histogram -4..4 (8 bins):", np.histogram(D.m0.clip(-4, 4), np.linspace(-4, 4, 9))[0])
print("   log10 sd histogram -1..1.5 (10 bins):", np.histogram(D.ls0.clip(-1, 1.5), np.linspace(-1, 1.5, 11))[0])
print("2) corr(mean, log sd) %.3f  corr(|mean|, log sd) %.3f" % (np.corrcoef(D.m0, D.ls0)[0, 1], np.corrcoef(D.m0.abs(), D.ls0)[0, 1]))
print("   label AUC: mean %.3f  |mean| %.3f  log sd %.3f" % (roc_auc_score(D.y, D.m0), roc_auc_score(D.y, D.m0.abs()), roc_auc_score(D.y, D.ls0)))
D["dl"] = np.log10((D.m1 - D.m0).abs() + 1e-12)
for lab in (0, 1):
    E = D[D.y == lab]
    print(f"3) label {lab}: slope of log|jump| on log total sd %.2f, on log noise sd %.2f;  corr %.3f / %.3f" % (
        np.polyfit(np.log10(E.s0), E.dl, 1)[0], np.polyfit(np.log10(E.z0), E.dl, 1)[0],
        np.corrcoef(np.log10(E.s0), E.dl)[0, 1], np.corrcoef(np.log10(E.z0), E.dl)[0, 1]))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].hist(D.m0.clip(-5, 5), 80); ax[0].set_title("pre mean")
ax[1].hist(D.ls0.clip(-1, 1.5), 80); ax[1].set_title("log10 pre sd")
ax[2].scatter(D.m0, D.ls0, s=1, alpha=.3); ax[2].set_xlim(-5, 5); ax[2].set_title("mean vs log10 sd")
plt.tight_layout(); plt.savefig("outputs/probe/base_check.png", dpi=90)
B = D[D.m0.abs() > 3]; C = D[D.m0.abs() <= 3]
print("4) |mean|>3 group: n %d, sd quantiles" % len(B), np.round(B.s0.quantile(q).to_numpy(), 2), " core sd", np.round(C.s0.quantile(q).to_numpy(), 2))
print("   |mean|/sd big group", np.round((B.m0.abs() / B.s0).quantile(q).to_numpy(), 2), " label rate big %.3f core %.3f" % (B.y.mean(), C.y.mean()))
print(f"n={len(D)}  {time.time() - t0:.1f}s")
