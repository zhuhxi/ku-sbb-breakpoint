"""Two doubts about the outer shell: (1) is the |mean|>3 group a plain rescale (same standardized shape)?
(2) is the size applied after the break (relative jump the same in big and core groups)?

    python probe/base_check2.py
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score

t0 = time.time()
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
rows = {}
for i, g in S.groupby("id", sort=False):
    v, p = g.value.to_numpy(), g.period.to_numpy()
    a, b = v[p == 0], v[p == 1]
    z = (a - a.mean()) / a.std()
    rows[i] = dict(n=len(v), frac=len(a) / len(v), m0=a.mean(), s0=a.std(), ac1=np.corrcoef(z[1:], z[:-1])[0, 1],
                   kurt=kurtosis(z), skew=skew(z), cv=a.std() / max(abs(a.mean()), 1e-9),
                   rjump=abs(b.mean() - a.mean()) / a.std(), rsd=abs(np.log(b.std() / a.std())))
D = pd.DataFrame.from_dict(rows, orient="index").join(y.rename("y"))
big = D.m0.abs() > 3
q = [.1, .25, .5, .75, .9]
print("1) standardized shape, big group vs core (quantiles 10/25/50/75/90); AUC big-vs-core")
for c in ["n", "frac", "ac1", "kurt", "skew"]:
    print(f"   {c:5s} big {np.round(D.loc[big, c].quantile(q).to_numpy(), 2)}  core {np.round(D.loc[~big, c].quantile(q).to_numpy(), 2)}  AUC {roc_auc_score(big, D[c]):.3f}")
print("2) relative jump |mean after - before| / sd before, and |log sd ratio|  (median label0 -> label1)")
for name, m in [("core", ~big), ("big", big)]:
    E = D[m]
    print(f"   {name:4s} rjump {E[E.y == 0].rjump.median():.3f} -> {E[E.y == 1].rjump.median():.3f}  (q75 {E[E.y == 1].rjump.quantile(.75):.3f})"
          f"  AUC {roc_auc_score(E.y, E.rjump):.3f} |  rsd {E[E.y == 0].rsd.median():.3f} -> {E[E.y == 1].rsd.median():.3f}  AUC {roc_auc_score(E.y, E.rsd):.3f}")
C = D[~big & (D.y == 1)]
for lo, hi in [(0.5, .8), (.8, 1.2), (1.2, 2), (2, 5)]:
    E = D[~big & D.s0.between(lo, hi)]
    print(f"   core sd {lo}-{hi}: n {len(E)}  rjump AUC {roc_auc_score(E.y, E.rjump):.3f}  label1 median rjump {E[E.y == 1].rjump.median():.3f}")
print(f"{time.time() - t0:.1f}s")
