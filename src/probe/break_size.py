"""How much do level, spread and dependence change at the boundary? Stationary series only (structure before the
boundary = WN / AR1 / AR2). For each series:
  level  = (mean after - mean before) / sd before
  spread = log(sd of AR(2) residuals after / before)
  dep    = lag-1 autocorrelation after - before
Break-series share with a change is estimated as excess over no-break series: P1(|s| > t) - P0(|s| > t).

    python probe/break_size.py --n 4000
"""
import argparse
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4000)
args = ap.parse_args()
sys.argv = [sys.argv[0], "x"]
spec = importlib.util.spec_from_file_location("cp", "probe/calib_params.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def ar2_resid_sd(x):
    X = np.column_stack([np.ones(len(x) - 2), x[1:-1], x[:-2]])
    b, *_ = np.linalg.lstsq(X, x[2:], rcond=None)
    return (x[2:] - X @ b).std()


def one(v, n0):
    a, b = v[:n0], v[n0:]
    ac = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]
    return dict(cls=cp.classify(a)["cls"], level=(b.mean() - a.mean()) / a.std(),
                spread=np.log(ar2_resid_sd(b) / ar2_resid_sd(a)), dep=ac(b) - ac(a))


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    r = np.random.default_rng(0)
    ids = np.r_[r.choice(lab.index[lab == 0], args.n, False), r.choice(lab.index[lab == 1], args.n, False)]
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(ids))].sort_values(["id", "time"])
    keys, rows = zip(*[(k, (g["value"].to_numpy(), int((g["period"] == 0).sum()))) for k, g in S.groupby("id")])
    D = pd.DataFrame(Parallel(n_jobs=30)(delayed(one)(v, n0) for v, n0 in rows), index=keys)
    D["y"] = lab.reindex(D.index).to_numpy()
    D = D[D.cls.isin(["WN", "AR1", "AR2"])]
    D.to_parquet("outputs/probe/break_size.parquet")
    print(f"stationary series: no-break {sum(D.y == 0)}, break {sum(D.y == 1)}   ({time.time() - t0:.1f}s)")
    for s in ["level", "spread", "dep"]:
        z0, z1 = D.loc[D.y == 0, s].abs(), D.loc[D.y == 1, s].abs()
        ex = {f"t=q{int(q * 100)}": round(float((z1 > z0.quantile(q)).mean() - (1 - q)), 3) for q in (.5, .75, .9, .99)}
        big = D.loc[(D.y == 1) & (D[s].abs() > z0.quantile(.99)), s]
        print(f"{s:7s} excess share of breaks with |{s}| > no-break quantile: {ex}")
        print(f"        no-break |{s}| quantiles 50/90/99%: {np.round(z0.quantile([.5, .9, .99]).to_numpy(), 3)}")
        print(f"        clearly changed ({len(big)}): size quantiles 10/50/90% of |{s}|: {np.round(big.abs().quantile([.1, .5, .9]).to_numpy(), 2)}, up {np.mean(big > 0):.2f}")
    print(f"total {time.time() - t0:.1f}s")
