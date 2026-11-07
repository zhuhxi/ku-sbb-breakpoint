"""Error analysis of the combo10 LGB OOF: what do missed breaks and false alarms look like?

    python probe/hard_cases.py

Groups (by OOF score of the single combo10 LGB, outputs/factory/confirm_combo10.npz):
  TP = label 1, score in top half of positives;  FN = label 1, bottom 25% of positives
  FP = label 0, top 10% of negatives;            TN = label 0, bottom half of negatives
For each series a few plain diagnostics (robust, before = b, after = a) are printed as group medians,
so we can see which kind of change the model misses. Writes outputs/probe/hard_cases.parquet.
"""
from pathlib import Path

import numpy as np
import pandas as pd

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
oof = np.load("outputs/factory/confirm_combo10.npz")["oof"]
S = pd.read_parquet("data/train_series.parquet")


def mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9


def acf1(v):
    v = v - v.mean()
    return (v[1:] * v[:-1]).sum() / ((v * v).sum() + 1e-12)


def diag(g):
    v, p = g["value"].to_numpy(), g["period"].to_numpy()
    b, a = v[p == 0], v[p == 1]
    sb = mad(b)
    th = np.array_split(a, 3)
    lev = [(np.median(t) - np.median(b)) / sb for t in th]
    db, da = np.diff(b), np.diff(a)
    return dict(
        n0=len(b), n1=len(a), nmin=min(len(b), len(a)),
        lev=(np.median(a) - np.median(b)) / sb,                       # level shift in before-MAD units
        lev_min3=np.sign(lev[0]) * min(abs(x) for x in lev) if np.all(np.sign(lev) == np.sign(lev[0])) else 0.0,
        lev_last=lev[2], lev_first=lev[0],
        lsd=np.log(mad(a) / sb),                                      # spread change
        ldsd=np.log(mad(da) / mad(db)),                               # spread change of increments
        dacf=acf1(a) - acf1(b),                                       # dependence change
        acf_b=acf1(b),
        spike_a=np.max(np.abs(a - np.median(a))) / mad(a),            # biggest outlier after
        spike_b=np.max(np.abs(b - np.median(b))) / sb,
        drift_b=(np.median(b[len(b) // 2:]) - np.median(b[:len(b) // 2])) / sb,  # natural drift inside before
    )


D = pd.DataFrame({i: diag(g) for i, g in S.groupby("id", sort=False)}).T.reindex(lab.index).astype(float)
D["y"], D["oof"] = lab.to_numpy(), oof
pos, neg = D[D.y == 1], D[D.y == 0]
grp = pd.Series("mid", index=D.index)
grp[pos.index[pos.oof >= pos.oof.quantile(0.5)]] = "TP"
grp[pos.index[pos.oof <= pos.oof.quantile(0.25)]] = "FN"
grp[neg.index[neg.oof >= neg.oof.quantile(0.9)]] = "FP"
grp[neg.index[neg.oof <= neg.oof.quantile(0.5)]] = "TN"
D["grp"] = grp
Path("outputs/probe").mkdir(parents=True, exist_ok=True)
D.to_parquet("outputs/probe/hard_cases.parquet")

cols = ["nmin", "lev", "lev_min3", "lev_first", "lev_last", "lsd", "ldsd", "dacf", "acf_b", "spike_a", "spike_b", "drift_b"]
A = D[D.grp != "mid"].assign(**{c: D[c].abs() for c in ["lev", "lev_min3", "lev_first", "lev_last", "lsd", "ldsd", "dacf", "drift_b"]})
pd.set_option("display.width", 200)
print("group medians (abs values for signed diagnostics)")
print(A.groupby("grp")[cols].median().loc[["TP", "FN", "FP", "TN"]].round(3))
print(A.grp.value_counts())
