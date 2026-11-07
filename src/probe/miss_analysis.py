"""What does the best model (combo10g OOF) miss? Among series the model scores low (OOF < 0.3), compare real breaks
(missed) with no-breaks on simple statistics: a statistic that still separates them there carries information the
model is not using. Also the same in the high band (false alarms vs caught breaks), and plots of the worst cases.

    python probe/miss_analysis.py
"""
import sys
import time

sys.argv = [sys.argv[0]]
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
hd = load("hd", "probe/hidden_diag.py")
mf = load("mf", "probe/model_feats.py")
bs = load("bs", "probe/break_size.py")


def feats(v, n0):
    f = hd.one(v, n0)
    f.update(mf.gfeats(v, n0))
    f["cls"] = bs.one(v, n0)["cls"]
    f["n"], f["frac"] = len(v), n0 / len(v)
    return f


t0 = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
oof = pd.Series(np.load("outputs/factory/confirm_combo10g.npz")["oof"], index=lab.index)
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
G = {i: (g["value"].to_numpy(), int((g["period"] == 0).sum())) for i, g in S.groupby("id")}
F = pd.DataFrame(Parallel(n_jobs=30, batch_size=128)(delayed(feats)(*G[i]) for i in lab.index), index=lab.index)
F.to_parquet("outputs/probe/miss_feats.parquet")
F["y"], F["oof"] = lab, oof
print(f"features done ({time.time() - t0:.0f}s)")
print(f"breaks with OOF < 0.3: {((F.y == 1) & (F.oof < .3)).mean() * 100:.1f}% of all series, "
      f"{(F.oof[F.y == 1] < .3).mean() * 100:.1f}% of the breaks")
num = [c for c in F.columns if c not in ("cls", "y", "oof")]
for band, m in [("low band (OOF < 0.3): missed breaks vs no-breaks", F.oof < .3),
                ("high band (OOF > 0.7): caught breaks vs false alarms", F.oof > .7)]:
    D = F[m]
    a = {c: roc_auc_score(D.y, D[c].fillna(0)) for c in num}
    a = pd.Series({c: max(v, 1 - v) for c, v in a.items()}).sort_values(ascending=False)
    print(f"\n{band}: {int(D.y.sum())} breaks / {int((1 - D.y).sum())} no-breaks; top separating statistics (AUC)")
    print("  " + "\n  ".join(f"{c:45s} {v:.3f}" for c, v in a.head(8).items()))
    print("  structure mix (% breaks / % no-breaks): " + "  ".join(
        f"{k} {(D[D.y == 1].cls == k).mean() * 100:.0f}/{(D[D.y == 0].cls == k).mean() * 100:.0f}" for k in D.cls.value_counts().index[:6]))
fig, ax = plt.subplots(4, 3, figsize=(15, 11))
pick = list(F[F.y == 1].nsmallest(6, "oof").index) + list(F[F.y == 0].nlargest(6, "oof").index)
for a, i in zip(ax.ravel(), pick):
    v, n0 = G[i]
    a.plot(v, lw=.6, color="0.4")
    a.plot(pd.Series(v).rolling(30, center=True).mean(), color="tab:blue", lw=1.2)
    a.axvline(n0, color="k", ls="--", lw=.8)
    a.set_title(f"{'MISSED break' if F.y[i] else 'FALSE alarm'}  oof {F.oof[i]:.2f}  {F.cls[i]}", fontsize=9)
fig.tight_layout()
fig.savefig("outputs/probe/miss_cases.png", dpi=80)
print(f"total {time.time() - t0:.0f}s")
