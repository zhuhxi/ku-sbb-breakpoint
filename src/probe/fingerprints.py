"""Code fingerprints on the real training series (no generator involved):
 1 boundary join: lag-1 correlation across the boundary pair vs inside the segments
 2 warm-up at the start: are the first points calmer (recursion started from 0)?
 3 warm-up after the boundary: same, for the first points after the boundary
 4 numeric precision: dtype, decimals
 5 round-number parameters: spikes in the pre-part lag-1 autocorrelation / sine period

    python probe/fingerprints.py [1-5 ...]
"""
import sys
import time

import numpy as np
import pandas as pd

which = sys.argv[1:] or list("12345")
t0 = time.time()
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
gs = pd.read_parquet("outputs/factory/genA.parquet")["g_struct"]
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
X = dict(zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut)))
N0 = S[S.period == 0].groupby("id").size()
print(f"loaded {len(X)} series {time.time() - t0:.1f}s")


def z(v):
    return (v - v.mean()) / v.std()


rows = []
for i, x in X.items():
    n0 = N0[i]
    a, b = z(x[:n0]), z(x[n0:])
    rows.append(dict(id=i, y=y[i], g=gs.get(i, -1), ac=np.mean(a[1:] * a[:-1]),
                     cross=a[-1] * b[0], inside=np.mean(a[-11:-1] * a[-10:]),   # last 10 pairs before the boundary
                     start=np.mean(a[:5] ** 2), later=np.mean(a[30:35] ** 2),
                     bstart=np.mean(b[:5] ** 2), blater=np.mean(b[30:35] ** 2)))
D = pd.DataFrame(rows).set_index("id")
dep = D[D.ac > .5]                                            # strongly dependent series show the effects best
print(f"measured {time.time() - t0:.1f}s, {len(dep)} series with lag-1 corr > .5")
if "1" in which:
    print("\n[1] boundary join (mean product of standardized neighbours; contiguous code -> cross ~ inside)")
    print(dep.groupby("y")[["ac", "inside", "cross"]].mean().round(3).to_string())
if "2" in which:
    print("\n[2] start warm-up: mean square of first 5 points vs points 30-34 (1 = no warm-up effect)")
    print(dep.groupby("y")[["start", "later"]].mean().round(3).to_string())
if "3" in which:
    print("\n[3] warm-up after the boundary: first 5 points after vs points 30-34 after")
    print(dep.groupby("y")[["bstart", "blater"]].mean().round(3).to_string())
if "4" in which:
    v = S.value.to_numpy()
    print("\n[4] dtype", S.value.dtype, " decimals needed (share of values exact at k decimals):",
          {k: round(float(np.mean(np.abs(np.round(v, k) - v) < 1e-12)), 3) for k in (2, 4, 6, 8, 10)})
    print("    distinct values", len(np.unique(v)), "of", len(v), "  float32-exact share", round(float(np.mean(v.astype(np.float32).astype(np.float64) == v)), 3))
if "5" in which:
    st = D[D.g.isin([1, 2])]                                  # fitted AR groups
    h = np.histogram(st.ac, np.linspace(-1, 1, 81))[0]
    print("\n[5] AR-group lag-1 corr histogram (bins of .025), spikes = count / neighbour mean:")
    sp = h / np.maximum((np.r_[h[1:], 0] + np.r_[0, h[:-1]]) / 2, 1)
    top = np.argsort(sp)[::-1][:5]
    print("    top spikes at", [(round(-1 + .025 * k + .0125, 3), int(h[k]), round(float(sp[k]), 2)) for k in top])
    per = []
    for i in D.index[D.g.isin([6, 7])]:
        a = X[i][:N0[i]]
        f = np.abs(np.fft.rfft(a - a.mean()))[1:]
        per.append(len(a) / (f.argmax() + 1))
    per = np.array(per)
    print("    SIN-group period: share within .1 of an integer", round(float(np.mean(np.abs(per - per.round()) < .1)), 3),
          "(no preference ~ .2);  counts by rounded period", pd.Series(per.round().astype(int)).value_counts().sort_index().to_dict())
print(f"\ntotal {time.time() - t0:.1f}s")
