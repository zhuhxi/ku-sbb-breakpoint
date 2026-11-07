"""Random sample of error cases (outputs/error_cases.csv) for eyeballing: 50 missed breaks and 50 false alarms,
one figure each (5 x 10). Panel: series, boundary (dotted), before/after means (dashed), rolling sd 30 (grey,
right axis), title = structure, n, score.

    python probe/error_gallery.py   # -> outputs/figs/err_missed.png, outputs/figs/err_false.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E = pd.read_csv("outputs/error_cases.csv", dtype={"id": str}).set_index("id")
pick = {k: E[E.kind == k].sample(50, random_state=1) for k in ("missed break", "false alarm")}
ids = set(pick["missed break"].index) | set(pick["false alarm"].index)
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
S = S[S.id.isin(ids)].sort_values(["id", "time"])
G = {k: g for k, g in S.groupby("id")}
for kind, fn, col in (("missed break", "err_missed", "#c0392b"), ("false alarm", "err_false", "#d35400")):
    P = pick[kind].sort_values("structure")
    fig, axes = plt.subplots(10, 5, figsize=(30, 34))
    for ax, (k, row) in zip(axes.ravel(), P.iterrows()):
        x = G[k]["value"].to_numpy()
        n0 = int(row.n0)
        t = np.arange(len(x))
        ax.plot(t, x, lw=.5, color=col)
        ax.axvline(n0, color="k", ls=":", lw=1.2)
        ax.hlines([x[:n0].mean()], 0, n0, colors="k", linestyles="--", lw=1)
        ax.hlines([x[n0:].mean()], n0, len(x), colors="k", linestyles="--", lw=1)
        ax2 = ax.twinx()
        ax2.plot(t, pd.Series(x).rolling(30, center=True).std(), color="grey", lw=.8, alpha=.8)
        ax2.set_yticks([])
        ax.set_title(f"{row.structure}  n={len(x)}  score {row.score:.2f}", fontsize=11, loc="left")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"50 random {kind.upper()}S of the current best (label {int(P.label.iloc[0])}); sorted by fitted structure. "
                 "dotted = boundary, dashed = before/after means, grey = rolling sd (30)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, .985])
    fig.savefig(f"outputs/figs/{fn}.png", dpi=60)
    print("saved", fn, P.structure.value_counts().to_dict())
