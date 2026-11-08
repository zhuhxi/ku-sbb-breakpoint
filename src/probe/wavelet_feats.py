"""Haar wavelet detail energy per scale (1..6) for pre and post segments, log ratios + robust (MAD) versions -> outputs/factory/wavfe.parquet (prefix wv_)"""
import os, numpy as np, pandas as pd, time
from joblib import Parallel, delayed
def haar(x, J=6):
    out = []; a = np.asarray(x, float)
    for j in range(J):
        if len(a) < 4: out.append((np.nan, np.nan)); continue
        m = len(a) // 2 * 2; d = (a[0:m:2] - a[1:m:2]) / np.sqrt(2); a = (a[0:m:2] + a[1:m:2]) / np.sqrt(2)
        out.append((np.log(np.mean(d ** 2) + 1e-12), np.log(np.median(np.abs(d)) + 1e-12)))
    return out
def one(sid, v, p):
    n0 = int((p == 0).sum()); A, B = haar(v[:n0]), haar(v[n0:]); f = {"id": sid}
    for j in range(6):
        f[f"wv_e{j}"] = B[j][0] - A[j][0]; f[f"wv_m{j}"] = B[j][1] - A[j][1]; f[f"wv_pre{j}"] = A[j][0]
    f["wv_slope_pre"] = A[3][0] - A[0][0]; f["wv_slope_post"] = B[3][0] - B[0][0]; f["wv_dslope"] = f["wv_slope_post"] - f["wv_slope_pre"]
    return f
t = time.time(); out = []
for part in ("train", "test"):
    S = pd.read_parquet(f"data/{part}_series.parquet").sort_values(["id", "time"], kind="stable")
    out += Parallel(n_jobs=30, batch_size=256)(delayed(one)(sid, g["value"].to_numpy(), g["period"].to_numpy()) for sid, g in S.groupby("id", sort=False))
F = pd.DataFrame(out).set_index("id"); F.to_parquet(os.environ.get("OUTW", "outputs/factory/wavfe.parquet")); print(F.shape, f"{time.time()-t:.0f}s")
