"""Targeted features from the panel review (probe/error_marks.csv):
(a) "should not flag": one spike exactly at n0 and nothing else  ->  spike indicator + the main before/after numbers
    recomputed with the boundary neighbourhood dropped and outliers clipped (so one point cannot drive them);
(b) "should catch": clean level steps in AR / ARMA series and sharp jumps at n0 in random walks  ->  robust step size
    in noise units, and the boundary increment against the series' own increments.
Robust scale everywhere = 1.4826 * MAD.
  tg_spike     max |z| over n0-2..n0+2 of AR(5) residuals (AR on the pre part), z by pre-residual MAD
  tg_spike_iso tg_spike minus the max |z| over n0+3..n0+10 (isolated spike = large)
  tg_dm_trim   (median after - median before) / MAD before, n0-2..n0+5 dropped, values clipped at 4 MAD
  tg_dv_trim   log(MAD of residual after / before), same trimming
  tg_dm_raw_minus_trim  |plain mean change| - |tg_dm_trim|  (how much of the level change is due to the boundary points)
  tg_step_res  median of AR(5) residuals over n0+3..n0+40, in pre-residual MAD units (lasting step, spike-free)
  tg_inc_z     |x_n0 - x_{n0-1}| / MAD of all increments ;  tg_inc_z3 = max over n0..n0+2
  tg_inc_rank  rank of |x_n0 - x_{n0-1}| among all increments

    python probe/target_feats.py     # train + test -> outputs/factory/target.parquet
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

P = 5
mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12


def one(x, n0):
    n = len(x)
    pre = x[:n0]
    A = np.column_stack([np.ones(n0 - P)] + [pre[P - k:n0 - k] for k in range(1, P + 1)])
    b = np.linalg.lstsq(A, pre[P:], rcond=None)[0]
    Af = np.column_stack([np.ones(n - P)] + [x[P - k:n - k] for k in range(1, P + 1)])
    e = np.r_[np.zeros(P), x[P:] - Af @ b]
    pe = e[P:n0]
    z = (e - np.median(pe)) / mad(pe)
    sp = np.abs(z[max(n0 - 2, P):n0 + 3]).max()
    after = np.abs(z[n0 + 3:n0 + 11]).max() if n - n0 > 12 else 0.0
    keep_b = np.r_[np.arange(0, max(n0 - 2, 1))]
    keep_a = np.arange(min(n0 + 6, n - 10), n)
    xb, xa = x[keep_b], x[keep_a]
    mb, sb = np.median(xb), mad(xb)
    xb, xa = np.clip(xb, mb - 4 * sb, mb + 4 * sb), np.clip(xa, mb - 4 * sb, mb + 4 * sb)
    eb, ea = e[P:max(n0 - 2, P + 5)], e[min(n0 + 6, n - 10):]
    dm_trim = (np.median(xa) - np.median(xb)) / sb
    d = np.abs(np.diff(x))
    md = mad(np.diff(x))
    return {"tg_adm_trim": abs(dm_trim), "tg_astep_res": abs(np.median(e[n0 + 3:n0 + 41]) / mad(pe)) if n - n0 > 45 else np.nan,
            "tg_spike": sp, "tg_spike_iso": sp - after, "tg_dm_trim": dm_trim, "tg_dv_trim": np.log(mad(ea) / mad(eb)),
            "tg_dm_raw_minus_trim": abs((x[n0:].mean() - pre.mean()) / pre.std()) - abs(dm_trim),
            "tg_step_res": np.median(e[n0 + 3:n0 + 41]) / mad(pe) if n - n0 > 45 else np.nan,
            "tg_inc_z": d[n0 - 1] / md, "tg_inc_z3": d[n0 - 1:n0 + 2].max() / md, "tg_inc_rank": float((d < d[n0 - 1]).mean())}


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    items = [(str(k), g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    rows = Parallel(n_jobs=30, batch_size=64)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    G.to_parquet("outputs/factory/target.parquet")
    print(f"{len(G)} series in {time.time() - t0:.1f}s")
