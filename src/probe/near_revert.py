"""Does the level blip right after the boundary in real NO-BREAK series revert (decoy) or persist (tiny unlabelled shift)?
Series flagged when 12 * mean(z_1..12)^2 > 3.84 (AR(5) one-step errors after the split). For them, signed raw shifts in
pre-sd units, sign = sign of the early mean error:
  early = mean x[c : c+12] - mean x[c-100 : c],   late = mean x[c+20 : c+120] - mean x[c-100 : c]
Real at n0 vs real at pseudo split n0 - n/10 (chance flags) vs fake at n0; real label-1 at n0 for reference.

    python probe/near_revert.py [N]
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


def one(x, c):
    if not (110 < c < len(x) - 125):
        return None
    b, e, _, _ = fa._arfit(x[:c], 5)
    z = fa._onestep(b, np.r_[x[c - 5:c], x[c:c + 12]], 5) / (e.std() + 1e-12)
    if 12 * z.mean() ** 2 <= 3.84:
        return None
    s, m0, sd = np.sign(z.mean()), x[c - 100:c].mean(), x[c - 100:c].std() + 1e-12
    return s * (x[c:c + 12].mean() - m0) / sd, s * (x[c + 20:c + 120].mean() - m0) / sd


def run(D, shift=False):
    R = [r for r in Parallel(n_jobs=30, batch_size=32)(delayed(one)(x, n0 - (len(x) // 10 if shift else 0)) for x, n0 in D) if r]
    R = np.array(R)
    return len(R) / len(D), np.median(R[:, 0]), np.median(R[:, 1])


L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
get = lambda lab: [(g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for _, g in
                   S[S.id.isin(set(L.loc[L.label == lab, "id"].to_numpy()[:N]))].sort_values(["id", "time"]).groupby("id", sort=False)]
real0, real1 = get(0), get(1)
fake = Parallel(n_jobs=30, batch_size=32)(delayed(cb.gen)(cb.BREAKS[MENU], 10_000 + s, False, True) for s in range(N))
for nm, D, sh in (("real 0 at n0", real0, False), ("real 0 pseudo", real0, True), ("fake 0 at n0", fake, False), ("real 1 at n0", real1, False)):
    f, a, b = run(D, sh)
    print(f"{nm:14s} flagged {f:.3f}  median early {a:+.3f}  late {b:+.3f}  late/early {b / a:+.2f}")
