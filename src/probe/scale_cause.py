"""Check the two guesses for the size gap (probe/scale_by_acf.py).
AR: generator uses unit innovations, so log sd grows with -0.5 log(1 - rho1^2); if the real generator normalised AR
series to unit size, real log sd would NOT grow with it. Slope of log sd on that term, series with rho1 in (-.9, -.2).
SIN: sine share of variance = power in the top periodogram peak (+-1 bin); quantiles over series where it is > .3
(real vs fake; fake amplitude U(.5, 3) on unit noise).

    python probe/scale_cause.py [N] [MENU]
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
MENU = sys.argv[2] if len(sys.argv) > 2 else "ok14"
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


def f(x):
    z = x - x.mean()
    P = np.abs(np.fft.rfft(z))[1:] ** 2
    k = P.argmax()
    return np.log(x.std()), z[1:] @ z[:-1] / (z @ z), P[max(k - 1, 0):k + 2].sum() / P.sum(), len(x) / (k + 1)


L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
S = pd.read_parquet("data/train_series.parquet")
S = S[S["id"].astype(str).isin(set(L.loc[L.label == 0, "id"].to_numpy()[:N]))].sort_values(["id", "time"])
cols = ["lsd", "acf", "pk", "per"]
R = pd.DataFrame([f(g["value"].to_numpy(np.float64)) for _, g in S.groupby("id", sort=False)], columns=cols)
F = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(lambda s: f(cb.gen(cb.BREAKS[MENU], 10_000 + s, False, True)[0]))(s) for s in range(N)), columns=cols)
for nm, D in (("real", R), ("fake", F)):
    a = D[(D.acf > -.9) & (D.acf < -.2)]
    g = -.5 * np.log(1 - a.acf ** 2)
    sl = np.polyfit(g, a.lsd, 1)[0]
    s = D[(D.pk > .3) & (D.per > 11) & (D.per < 34)]
    print(f"{nm}: AR (n={len(a)}) slope of log sd on -0.5log(1-rho^2) = {sl:+.2f} (unit innovations -> ~1, normalised -> ~0) | "
          f"SIN-like n={len(s)} share {len(s) / len(D):.3f}, sine share of variance q25/50/75 "
          + " ".join(f"{v:.2f}" for v in s.pk.quantile([.25, .5, .75])) + f", median log sd {s.lsd.median():+.3f}")
