"""Per-series break sizes on real white-noise-like series (pre and post lag-1 |corr| < .1, no RW):
level jump in pre-sd units, log spread ratio. Label 1 histogram minus label 0 histogram = what breaks add.

    python probe/break_sizes_real.py
"""
import time

import numpy as np
import pandas as pd

t0 = time.time()
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id")["label"]
ac = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]
rows = {}
for i, g in S.groupby("id", sort=False):
    v, p = g.value.to_numpy(), g.period.to_numpy()
    a, b = v[p == 0], v[p == 1]
    if abs(ac(a)) < .1 and abs(ac(b)) < .1:
        rows[i] = dict(level=(b.mean() - a.mean()) / a.std(), spread=np.log(b.std() / a.std()), n=len(v))
D = pd.DataFrame.from_dict(rows, orient="index").join(y)
print(f"white-noise-like series: {len(D)} (label1 {int(D.label.sum())})")
for c, edges in [("level", np.r_[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 5]),
                 ("spread", np.r_[0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .7, .8, 1, 1.5])]:
    h1 = np.histogram(D.loc[D.label == 1, c].abs(), edges)[0]
    h0 = np.histogram(D.loc[D.label == 0, c].abs(), edges)[0] * (D.label == 1).sum() / (D.label == 0).sum()
    print(f"\n|{c}| bin       label1  label0(scaled)  excess")
    for k in range(len(edges) - 1):
        print(f"  {edges[k]:4.2f}-{edges[k+1]:4.2f}   {h1[k]:6d}   {h0[k]:8.0f}     {h1[k] - h0[k]:+6.0f}")
print(f"{time.time() - t0:.1f}s")
