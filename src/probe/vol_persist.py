"""Do volatility changes at a real break persist to the end, and do no-break series show lasting changes too?
Per series (AR(5) residuals of the whole series): r_early = log sd(first third after n0) - log sd(pre),
r_late = log sd(last third after n0) - log sd(pre). Rows: |r_early| bins; columns: label 0 / 1 counts and the median
r_late / r_early ("how much is left at the end": 1 = persists, 0 = gone). Control: the same at a pseudo split
inside the pre part of label-0 series (natural volatility drift).

    python probe/vol_persist.py
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def res(x, p=5):
    A = np.column_stack([np.ones(len(x) - p)] + [x[p - k:len(x) - k] for k in range(1, p + 1)])
    return x[p:] - A @ np.linalg.lstsq(A, x[p:], rcond=None)[0]


def one(x, n0):
    e = res(x)
    c = n0 - 5
    pre, post = e[:c], e[c:]
    m = len(post) // 3
    out = dict(re=np.log(post[:m].std() / pre.std()), rl=np.log(post[-m:].std() / pre.std()))
    if c > 300:                                   # pseudo split in the middle of the pre part (no boundary there)
        h = c // 2
        a, b = pre[:h], pre[h:]
        mb = len(b) // 3
        out.update(pe=np.log(b[:mb].std() / a.std()), pl=np.log(b[-mb:].std() / a.std()))
    return out


L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
S = S.sort_values(["id", "time"])
items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
R = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(one)(x, n0) for _, x, n0 in items), index=[k for k, _, _ in items])
R["y"] = L.loc[R.index].to_numpy()
bins = [0, .15, .3, .5, .8, 5]
R["b"] = pd.cut(R.re.abs(), bins)
rows = []
for b, g in R.groupby("b", observed=True):
    r = {"|log sd change| early after n0": str(b)}
    for lab in (0, 1):
        h = g[g.y == lab]
        r[f"n label {lab}"] = len(h)
        r[f"left at end, label {lab}"] = round(float(np.median(h.rl / h.re)), 2)
    r["share label 1"] = round(float((g.y == 1).mean()), 3)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
P = R[(R.y == 0) & R.pe.notna()]
P = P.assign(b=pd.cut(P.pe.abs(), bins))
print("\ncontrol: pseudo split inside the pre part of label-0 series (no boundary)")
print(P.groupby("b", observed=True).apply(lambda g: pd.Series({"n": len(g), "left at end": round(float(np.median(g.pl / g.pe)), 2)})).to_string())
