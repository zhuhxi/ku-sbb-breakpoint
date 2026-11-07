"""Gallery of the series the current best gets wrong, for eyeballing (current best = LGB + inter arms 1, 2, OOF).
Rows: missed breaks (label 1, lowest scores), false alarms (label 0, highest scores), then typical caught breaks and
typical no-break series. Each panel: series, boundary, before/after means (dashed), rolling sd (grey, right axis),
points beyond 4 robust sd of the pre-boundary part marked (official note: temporary spikes occur without a break).

    python probe/miss_gallery.py   # -> outputs/figs/miss_gallery.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

STRUCT = ["WN", "AR1", "AR2", "MA1", "ARMA11", "RW", "SIN+WN", "SIN+AR1"]
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = L.to_numpy()
r = lambda v: rankdata(v) / len(v)
I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
s = pd.Series((r(np.load("outputs/factory/confirm_combo10gnj.npz")["oof"]) + r(I[1]) + r(I[2])) / 3, index=L.index)
A = pd.read_parquet("outputs/factory/genA.parquet")
A.index = A.index.astype(str)
st = A["g_struct"].reindex(L.index).map(dict(enumerate(STRUCT)))
pos, neg = s[y == 1].sort_values(), s[y == 0].sort_values()
rows = [("MISSED break (label 1, lowest score)", pos.index[:6]),
        ("FALSE ALARM (label 0, highest score)", neg.index[::-1][:6]),
        ("caught break (label 1, median of top half)", pos.index[int(len(pos) * .85):][:6]),
        ("typical no-break (label 0, median score)", neg.index[len(neg) // 2:][:6])]
ids = {i for _, g in rows for i in g}
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
S = S[S.id.isin(ids)].sort_values(["id", "time"])
G = {k: g for k, g in S.groupby("id")}
fig, axes = plt.subplots(4, 6, figsize=(30, 15))
colors = ["#c0392b", "#d35400", "#2471a3", "#566573"]
for ri, (title, sel) in enumerate(rows):
    for ci, k in enumerate(sel):
        ax = axes[ri, ci]
        g = G[k]
        x = g["value"].to_numpy()
        n0 = int((g["period"] == 0).sum())
        t = np.arange(len(x))
        ax.plot(t, x, lw=.6, color=colors[ri])
        ax.axvline(n0, color="k", ls=":", lw=1.2)
        ax.hlines([x[:n0].mean()], 0, n0, colors="k", linestyles="--", lw=1)
        ax.hlines([x[n0:].mean()], n0, len(x), colors="k", linestyles="--", lw=1)
        med, mad = np.median(x[:n0]), np.median(np.abs(x[:n0] - np.median(x[:n0]))) * 1.4826 + 1e-12
        sp = np.abs(x - med) > 4 * mad
        ax.scatter(t[sp], x[sp], s=8, color="k", zorder=3)
        ax2 = ax.twinx()
        ax2.plot(t, pd.Series(x).rolling(30, center=True).std(), color="grey", lw=.8, alpha=.7)
        ax2.set_yticks([])
        ax.set_title(f"{st[k]}  n={len(x)}  score {s[k]:.2f}\n{'pre' if ci else title}" if ci == 0 else f"{st[k]}  n={len(x)}  score {s[k]:.2f}",
                     fontsize=10, loc="left")
        ax.tick_params(labelsize=7)
    axes[ri, 0].set_ylabel(title, fontsize=11, color=colors[ri])
fig.suptitle("Where the current best (OOF 0.8566) goes wrong. Dotted = boundary, dashed = before/after means, grey = rolling sd (30), dots = |x - pre median| > 4 robust sd",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .97])
fig.savefig("outputs/figs/miss_gallery.png", dpi=80)
print("saved")
