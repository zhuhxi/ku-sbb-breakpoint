"""Reverse-engineer the generator, step 1: what does a no-break series look like, and what changes at a break?"""
import numpy as np, pandas as pd
from scipy import stats
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
resc = pd.read_parquet("outputs/magic_b_train.parquet")["mb_rescaled"].reindex(lab.index).fillna(0)
d = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])

def ar(x, p=5):
    X = np.column_stack([np.ones(len(x) - p)] + [x[p - i:-i] for i in range(1, p + 1)])
    b, *_ = np.linalg.lstsq(X, x[p:], rcond=None)
    e = x[p:] - X @ b
    return b, e

def seg(x):
    t = np.arange(len(x)); sl = np.polyfit(t, x, 1)[0]
    b, e = ar(x, 5)
    return dict(mean=x.mean(), std=x.std(), slope=sl * len(x) / (x.std() + 1e-12), phi1=b[1], phi2=b[2], phi_sum=b[1:].sum(),
                rstd=e.std(), rkurt=stats.kurtosis(e), rskew=stats.skew(e),
                arch=np.corrcoef(e[1:] ** 2, e[:-1] ** 2)[0, 1], acf1=np.corrcoef(x[1:], x[:-1])[0, 1],
                dacf1=np.corrcoef(np.diff(x)[1:], np.diff(x)[:-1])[0, 1])

rows = []
for sid, g in d.groupby("id", sort=False):
    v, p = g["value"].to_numpy(), g["period"].to_numpy()
    a, b = seg(v[p == 0]), seg(v[p == 1])
    r = {"id": sid, **{f"{k}_0": a[k] for k in a}, **{f"{k}_1": b[k] for k in b}}
    rows.append(r)
S = pd.DataFrame(rows).set_index("id").loc[lab.index]
S["label"], S["resc"] = lab, resc
S.to_parquet("outputs/gen_segstats.parquet")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
K = ["mean", "std", "slope", "phi1", "phi2", "phi_sum", "rkurt", "rskew", "arch", "acf1", "dacf1"]
N = S[(S.label == 0) & (S.resc == 0)]
print("== no-break series, BEFORE segment: parameter distribution (quantiles)")
print(N[[f"{k}_0" for k in K]].quantile([.01, .1, .25, .5, .75, .9, .99]).round(3).T)
print("\n== change after-before: quantiles by label (non-rescaled)")
for k in ["mean", "std", "phi1", "phi_sum", "rkurt", "slope", "acf1"]:
    if k == "std":
        dd = np.log(S["std_1"] / S["std_0"])
    else:
        dd = S[f"{k}_1"] - S[f"{k}_0"]
    q = dd[S.resc == 0].groupby(S.label).quantile([.02, .1, .25, .5, .75, .9, .98]).unstack().round(3)
    print(k, "(log ratio)" if k == "std" else "", "\n", q)
