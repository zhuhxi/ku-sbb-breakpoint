"""What changes at the boundary in break series? For each series compare the segment before and after the official
boundary on mean, variance and lag-1 autocorrelation. No-break series give the 'nothing happened' range (1%-99%);
a break series counts as changed in a quantity when it falls outside that range.

    python probe/break_what.py --n 3000
"""
import argparse
import time

import importlib.util
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000)
args = ap.parse_args()
sys.argv = [sys.argv[0], "x"]
spec = importlib.util.spec_from_file_location("cp", "probe/calib_params.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def seg_stats(v, n0):
    a, b = v[:n0], v[n0:]
    da, db = np.diff(a), np.diff(b)                         # differences: level shifts don't leak into variance
    sd = np.sqrt((da.var() + db.var()) / 4)
    ac = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]
    return dict(cls=cp.classify(a)["cls"],mean=(b.mean() - a.mean()) / sd / np.sqrt(1 / len(a) + 1 / len(b)) / np.sqrt(min(len(a), len(b))),
                logvar=np.log(db.var() / da.var()),
                acf1=ac(b) - ac(a))


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    r = np.random.default_rng(0)
    ids = np.r_[r.choice(lab.index[lab == 0], args.n, False), r.choice(lab.index[lab == 1], args.n, False)]
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(ids))].sort_values(["id", "time"])
    keys, rows = [], []
    for k, g in S.groupby("id"):
        keys.append(k)
        rows.append((g["value"].to_numpy(), int((g["period"] == 0).sum())))
    D = pd.DataFrame(Parallel(n_jobs=30)(delayed(seg_stats)(v, n0) for v, n0 in rows), index=keys)
    D["y"] = lab.reindex(D.index).to_numpy()
    Q = ["mean", "logvar", "acf1"]
    print("pre-boundary structure, break vs no-break:", pd.crosstab(D.cls, D.y, normalize="columns").round(3).to_dict())
    out = pd.DataFrame(False, index=D.index, columns=Q)
    for c, G in D.groupby("cls"):                           # 'nothing happened' range within the same structure
        lo, hi = G.loc[G.y == 0, Q].quantile(0.01), G.loc[G.y == 0, Q].quantile(0.99)
        out.loc[G.index] = (G[Q] < lo) | (G[Q] > hi)
    B = out[D.y == 1]
    for c in ["WN", "AR1", "SIN", "RW-like", "ARMA11"]:
        m = (D.cls[D.y == 1] == c).to_numpy()
        print(f"  {c:8s} n={m.sum():4d} changed:", B[m].mean().round(2).to_dict(), " nothing visible", round(float((~B[m].any(axis=1)).mean()), 2))
    print("share of series outside the no-break 1-99% range (no-break: 2% by construction)")
    print("  break series:", B.mean().round(3).to_dict())
    pat = B.apply(lambda s: "+".join(c for c in B.columns if s[c]) or "nothing visible", axis=1)
    print("  break series by pattern:", pat.value_counts(normalize=True).round(3).to_dict())
    print("  direction among changed:", {c: f"up {np.mean(D.loc[B.index[B[c]], c] > 0):.2f}" for c in B.columns})
    print(f"total {time.time() - t0:.1f}s")
