"""Did the noise TYPE change at the boundary? (generator finding: ~35% of breaks redraw the noise type.)
Residuals of the series' own structure (whole-series MLE fit, probe/gen_feats.py) are split at the boundary.
  same : one noise family for both sides, each side with its own scale (fit on residuals divided by each side's sd)
  own  : each side picks its own family (normal / t / bimodal, by BIC) and parameters
g_noise_lr = loglik(own) - loglik(same)  (scale changes are left to g_scale_lr); g_noise_diff = 1 if the two sides pick
different families; g_noise_b / g_noise_a = family index per side.

    python probe/noise_lr.py --jobs 64     # all train + test series -> outputs/factory/gnoise.parquet
"""
import argparse
import importlib.util
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=30)
ap.add_argument("--n", type=int, default=0)   # >0: only the first n series (timing check)
args = ap.parse_args()
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("cl", "probe/close_loop.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)
NOISE = ["normal", "t", "bimodal"]
bic = lambda r, n: -2 * r[0] + r[1] * np.log(n)


def best(e):
    _, _, out = cl.fit_noise(e, full=True)
    k = min(out, key=lambda m: bic(out[m], len(e)))
    return k, out[k][0]


def one(v, n0):
    _, win, rw = cl.mle.fit("w", v)
    e = rw[win]["e"]
    k = max(n0 - cl.mle.M, 5)
    eb, ea = e[:k], e[k:]
    if len(ea) < 10:
        return {}
    sb, sa = eb.std() + 1e-12, ea.std() + 1e-12
    kj, llj = best(np.r_[eb / sb, ea / sa])
    llj -= len(eb) * np.log(sb) + len(ea) * np.log(sa)        # back to the residual scale
    kb, llb = best(eb)
    ka, lla = best(ea)
    return {"g_noise_lr": llb + lla - llj, "g_noise_diff": float(kb != ka), "g_noise_b": NOISE.index(kb), "g_noise_a": NOISE.index(ka)}


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    if args.n:
        items = items[: args.n]
    rows = Parallel(n_jobs=args.jobs, batch_size=16)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    if not args.n:
        G.to_parquet("outputs/factory/gnoise.parquet")
    print(f"{len(G)} series in {time.time() - t0:.1f}s\n{G.describe().round(3)}")
