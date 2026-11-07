"""Few series with a big blip, or all series a little?  Level or variance?  (no-break series, lags 1..12 after the split)
A = mean z^2 (one-step AR(5) errors / in-sample sd); level part Lv = 12 * mean(z)^2 (chi2_1 under no change);
variance part Va = var(z). Each at the true boundary vs the mean of two pseudo boundaries n0 +- n/10.

    python probe/near_kind.py [N]
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
MENU = sys.argv[2] if len(sys.argv) > 2 else "ok4"
sys.argv = [sys.argv[0]]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
fa = load("fa", "factory.py")
H = 12


def z(x, c, p=5):
    b, e, _, _ = fa._arfit(x[:c], p)
    return fa._onestep(b, np.r_[x[c - p:c], x[c:c + H]], p) / (e.std() + 1e-12)


def st(q):
    return np.array([np.mean(q * q), H * q.mean() ** 2, q.var()])


def one(x, n0):
    n, s = len(x), max(len(x) // 10, 25)
    ps = [st(z(x, c)) for c in (n0 - s, n0 + s) if 65 < c < n - 25]
    return np.r_[st(z(x, n0)), np.mean(ps, 0) if ps else np.full(3, np.nan)]


L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
ids = set(L.loc[L.label == 0, "id"].to_numpy()[:N])
S = pd.read_parquet("data/train_series.parquet")
S = S[S["id"].astype(str).isin(ids)].sort_values(["id", "time"])
real = [(g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for _, g in S.groupby("id", sort=False)]
fake = Parallel(n_jobs=30, batch_size=32)(delayed(cb.gen)(cb.BREAKS[MENU], 10_000 + s, False, True) for s in range(N))
for nm, D in (("real", real), ("fake", fake)):
    R = np.array(Parallel(n_jobs=30, batch_size=32)(delayed(one)(x, n0) for x, n0 in D))
    R = R[~np.isnan(R).any(1)]
    lr = np.log(R[:, 0] / R[:, 3])
    print(f"{nm}: all z^2 at/pseudo  mean {R[:, 0].mean() / R[:, 3].mean():.2f}  median {np.median(R[:, 0]) / np.median(R[:, 3]):.2f}  "
          f"| log ratio q25 {np.quantile(lr, .25):+.2f} q50 {np.quantile(lr, .5):+.2f} q75 {np.quantile(lr, .75):+.2f} q95 {np.quantile(lr, .95):+.2f}")
    print(f"      level part at/pseudo mean {R[:, 1].mean() / R[:, 4].mean():.2f}  share level>chi2 95% (3.84): at {(R[:, 1] > 3.84).mean():.3f} pseudo {(R[:, 4] > 3.84).mean():.3f}"
          f" | variance part mean {R[:, 2].mean() / R[:, 5].mean():.2f} median {np.median(R[:, 2]) / np.median(R[:, 5]):.2f}")
