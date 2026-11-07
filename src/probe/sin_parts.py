"""Sine series too big: is it the sine amplitude A or the noise sd? Fit A sin + B cos at the periodogram peak (refined on
a fine grid around it), residual sd = noise. SIN-like series: top-peak share > .3, period 11-34. Medians of log A,
log noise sd, log(A / noise), real label 0 vs generated no-break (shell scale is in both A and noise alike).

    python probe/sin_parts.py [N] [MENU]
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
MENU = sys.argv[2] if len(sys.argv) > 2 else "ok15"
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


def f(x):
    n = len(x)
    z = x - x.mean()
    P = np.abs(np.fft.rfft(z))[1:] ** 2
    k = P.argmax()
    pk, per = P[max(k - 1, 0):k + 2].sum() / P.sum(), n / (k + 1)
    if not (pk > .3 and 11 < per < 34):
        return None
    t = np.arange(n)
    best = None
    for T in np.linspace(per * .97, per * 1.03, 31):
        A = np.column_stack([np.ones(n), np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T)])
        c, res, *_ = np.linalg.lstsq(A, x, rcond=None)
        r = x - A @ c
        if best is None or r @ r < best[0]:
            best = (r @ r, np.hypot(c[1], c[2]), r.std(), T)
    return np.log(best[1]), np.log(best[2]), best[3], np.log(x.std())


L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
S = pd.read_parquet("data/train_series.parquet")
S = S[S["id"].astype(str).isin(set(L.loc[L.label == 0, "id"].to_numpy()[:N]))].sort_values(["id", "time"])
cols = ["lA", "lsig", "T", "lsd"]
R = pd.DataFrame([r for r in Parallel(n_jobs=30, batch_size=32)(delayed(f)(g["value"].to_numpy(np.float64)) for _, g in S.groupby("id", sort=False)) if r], columns=cols)
F = pd.DataFrame([r for r in Parallel(n_jobs=30, batch_size=32)(delayed(lambda s: f(cb.gen(cb.BREAKS[MENU], 10_000 + s, False, True)[0]))(s) for s in range(N)) if r], columns=cols)
q = [.1, .25, .5, .75, .9]
for c, nm in (("lA", "log amplitude"), ("lsig", "log noise sd"), ("lsd", "log total sd")):
    print(f"{nm:14s} real " + " ".join(f"{v:+.3f}" for v in R[c].quantile(q)) + "   fake " + " ".join(f"{v:+.3f}" for v in F[c].quantile(q)))
for nm, D in (("real", R), ("fake", F)):
    d = D.lA - D.lsig
    print(f"{nm}: n {len(D)}  log(A/noise) q10/50/90 " + " ".join(f"{v:+.3f}" for v in d.quantile([.1, .5, .9]))
          + f"  corr(log A, log noise) {np.corrcoef(D.lA, D.lsig)[0, 1]:+.2f}  period q10/50/90 " + " ".join(f"{v:.1f}" for v in D["T"].quantile([.1, .5, .9])))
