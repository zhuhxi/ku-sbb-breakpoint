"""The same 100 random error cases as probe/error_gallery.py (random_state 1), drawn 25 per page, larger, with
numbers in each title for a panel-by-panel review:
  dm   = (mean after - mean before) / sd before              (level change, raw values)
  dv   = log sd after - log sd before of AR(5) residuals     (spread change; AR fitted on the pre part)
  da   = lag-1 autocorrelation after - before                 (dependence change)
  keep = late-third / early-third spread change after n0     (1 = lasting, 0 = gone)
Panels are numbered M01..M50 (missed breaks) and F01..F50 (false alarms) in structure order.
With a marks file (probe/error_marks.csv: panel, mark, note) the panels get a coloured frame.

    python probe/error_panels.py            # -> outputs/figs/panels_{M,F}{1,2}.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E = pd.read_csv("outputs/error_cases.csv", dtype={"id": str}).set_index("id")
pick = {k: E[E.kind == k].sample(50, random_state=1).sort_values("structure") for k in ("missed break", "false alarm")}
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
S = S[S.id.isin(set(pick["missed break"].index) | set(pick["false alarm"].index))].sort_values(["id", "time"])
G = {k: g["value"].to_numpy() for k, g in S.groupby("id")}
marks = pd.read_csv("probe/error_marks.csv").set_index("panel") if os.path.exists("probe/error_marks.csv") else None
MC = {"should catch": "#1e8449", "should not flag": "#1f618d", "ambiguous": "#b7950b", "invisible": "#7f8c8d", "looks like break": "#8e44ad"}


def nums(x, n0):
    pre, post = x[:n0], x[n0:]
    P = 5
    A = np.column_stack([np.ones(n0 - P)] + [pre[P - k:n0 - k] for k in range(1, P + 1)])
    b = np.linalg.lstsq(A, pre[P:], rcond=None)[0]
    Af = np.column_stack([np.ones(len(x) - P)] + [x[P - k:len(x) - k] for k in range(1, P + 1)])
    e = x[P:] - Af @ b
    ea, eb = e[:n0 - P], e[n0 - P:]
    m = len(eb) // 3
    ac = lambda s: np.corrcoef(s[1:], s[:-1])[0, 1]
    dv = np.log(eb.std() / ea.std())
    keep = np.log(eb[-m:].std() / ea.std()) / (np.log(eb[:m].std() / ea.std()) + 1e-9)
    return (post.mean() - pre.mean()) / pre.std(), dv, ac(post) - ac(pre), keep


for tag, kind, col in (("M", "missed break", "#c0392b"), ("F", "false alarm", "#d35400")):
    P = pick[kind]
    for page in (0, 1):
        fig, axes = plt.subplots(5, 5, figsize=(26, 18))
        for j, ax in enumerate(axes.ravel()):
            i = page * 25 + j
            k, row = P.index[i], P.iloc[i]
            x = G[k]
            n0 = int(row.n0)
            t = np.arange(len(x))
            ax.plot(t, x, lw=.6, color=col)
            ax.axvline(n0, color="k", ls=":", lw=1.3)
            ax.hlines([x[:n0].mean()], 0, n0, colors="k", linestyles="--", lw=1.1)
            ax.hlines([x[n0:].mean()], n0, len(x), colors="k", linestyles="--", lw=1.1)
            ax2 = ax.twinx()
            ax2.plot(t, pd.Series(x).rolling(30, center=True).std(), color="grey", lw=.9, alpha=.8)
            ax2.set_yticks([])
            dm, dv, da, keep = nums(x, n0)
            pid = f"{tag}{i + 1:02d}"
            ax.set_title(f"{pid} {row.structure} n={len(x)} s={row.score:.2f} | dm {dm:+.2f} dv {dv:+.2f} da {da:+.2f} keep {keep:.2f}",
                         fontsize=10.5, loc="left")
            ax.tick_params(labelsize=7)
            if marks is not None and pid in marks.index:
                for sp in ax.spines.values():
                    sp.set_edgecolor(MC[marks.loc[pid, "mark"]]); sp.set_linewidth(4)
                ax.text(.01, .97, marks.loc[pid, "mark"], transform=ax.transAxes, va="top", fontsize=11,
                        color=MC[marks.loc[pid, "mark"]], weight="bold")
        fig.suptitle(f"{kind.upper()}S {page * 25 + 1}-{page * 25 + 25} (label {int(P.label.iloc[0])}). dotted = boundary, dashed = means, grey = rolling sd",
                     fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, .975])
        fig.savefig(f"outputs/figs/panels_{tag}{page + 1}.png", dpi=55)
print("saved")
