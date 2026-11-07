"""Fakes look rougher (mean |diff| 1.14 vs 1.04): is it the overall scale or the shape? log mean|diff| = log sd + log(mean|diff| / sd).
Quantiles of each part and of the lag-1 autocorrelation, real label 0 vs generated no-break.

    python probe/rough_split.py [N] [MENU]
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
MENU = sys.argv[2] if len(sys.argv) > 2 else "ok14"
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


def f(x):
    d, sd = np.diff(x), x.std()
    z = x - x.mean()
    return np.log(np.abs(d).mean()), np.log(sd), np.abs(d).mean() / sd, z[1:] @ z[:-1] / (z @ z)


L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
S = pd.read_parquet("data/train_series.parquet")
S = S[S["id"].astype(str).isin(set(L.loc[L.label == 0, "id"].to_numpy()[:N]))].sort_values(["id", "time"])
real = np.array([f(g["value"].to_numpy(np.float64)) for _, g in S.groupby("id", sort=False)])
fake = np.array(Parallel(n_jobs=30, batch_size=32)(delayed(lambda s: f(cb.gen(cb.BREAKS[MENU], 10_000 + s, False, True)[0]))(s) for s in range(N)))
q = [.05, .1, .25, .5, .75, .9, .95]
for i, nm in enumerate(("log mean|diff|", "log sd (scale)", "mean|diff|/sd (shape)", "lag-1 acf")):
    print(f"{nm:22s} real " + " ".join(f"{v:+.3f}" for v in np.quantile(real[:, i], q)))
    print(f"{'':22s} fake " + " ".join(f"{v:+.3f}" for v in np.quantile(fake[:, i], q)))
