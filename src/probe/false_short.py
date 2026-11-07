"""Plot the most confident false alarms among short series (label 0, total length < 500): raw series with the
boundary, segment means (red) and +-1 sd bands, plus a rolling-mean (black) and rolling-sd strip below.

    python probe/false_short.py [K]   # -> outputs/probe/false_short.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
g = pd.Series(np.load("outputs/factory/confirm_combo10g.npz")["oof"], index=lab.index)
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
L = S.groupby("id").size().reindex(lab.index)
pick = g[(lab == 0) & (L < 500)].sort_values(ascending=False).head(K)
rank = g.rank(pct=True)
G = {k: v for k, v in S[S.id.isin(pick.index)].groupby("id")}
fig, ax = plt.subplots(K // 4 * 2, 4, figsize=(20, K // 4 * 5), gridspec_kw=dict(height_ratios=[3, 1] * (K // 4)))
for i, sid in enumerate(pick.index):
    d = G[sid]
    x = d["value"].to_numpy()
    n0 = int((d["period"] == 0).sum())
    a, b = ax[i // 4 * 2, i % 4], ax[i // 4 * 2 + 1, i % 4]
    a.plot(x, lw=.5, color="tab:blue")
    for lo, hi in ((0, n0), (n0, len(x))):
        m, s = x[lo:hi].mean(), x[lo:hi].std()
        a.hlines([m], lo, hi, color="red")
        a.fill_between([lo, hi], m - s, m + s, color="red", alpha=.12)
    a.axvline(n0, color="k", ls="--")
    rm = pd.Series(x).rolling(25, center=True).mean()
    a.plot(rm, color="k", lw=1)
    b.plot(pd.Series(x).rolling(25, center=True).std(), color="tab:green")
    b.axvline(n0, color="k", ls="--")
    s0, s1 = x[:n0].std(), x[n0:].std()
    a.set_title(f"#{i} n={len(x)} n0={n0} score top {100 * (1 - rank[sid]):.1f}%  dm={(x[n0:].mean() - x[:n0].mean()) / s0:+.2f}sd  sd x{s1 / s0:.2f}", fontsize=9)
fig.tight_layout()
fig.savefig("outputs/probe/false_short.png", dpi=70)
print(pick.round(3).to_string())
