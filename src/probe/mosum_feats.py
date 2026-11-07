"""MOSUM-style sliding-window differences (moving-sum change-point statistics), following the classical steps:
  1. whiten: AR(P) fitted on the pre part only (break-free reference), applied to the whole series -> residuals e
  2. robust scale: sigma = 1.4826 * MAD of the pre residuals, z = e / sigma
  3. at EVERY position t, adjacent windows [t-w, t) and [t, t+w), standardised so that the null is ~N(0, 1):
       lvl(t) = (mean right - mean left) * sqrt(w / 2)
       vol(t) = log(var right / var left) / sqrt(4 / (w - 1))
  4. several widths w in WS
  5. compare the boundary with the rest of the curve: at = |D(n0)|, rank = share of t with |D| < at,
     rel = at / max |D|, near = max |D| within w of n0 (delayed changes), dist = |argmax - n0| / n
  6. level and spread kept apart; multi-scale summaries over w (max of at, mean of rank).

    python probe/mosum_feats.py     # train + test -> outputs/factory/mosum.parquet
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

P, WS = 5, (10, 20, 40, 80)


def curves(z, w):
    n = len(z)
    c1, c2 = np.r_[0, np.cumsum(z)], np.r_[0, np.cumsum(z * z)]
    t = np.arange(w, n - w + 1)
    mL, mR = (c1[t] - c1[t - w]) / w, (c1[t + w] - c1[t]) / w
    vL = (c2[t] - c2[t - w]) / w - mL ** 2
    vR = (c2[t + w] - c2[t]) / w - mR ** 2
    lvl = (mR - mL) * np.sqrt(w / 2)
    vol = np.log((vR + 1e-12) / (vL + 1e-12)) / np.sqrt(4 / (w - 1))
    return t, lvl, vol


def one(x, n0):
    n = len(x)
    pre = x[:n0]
    A = np.column_stack([np.ones(n0 - P)] + [pre[P - k:n0 - k] for k in range(1, P + 1)])
    b = np.linalg.lstsq(A, pre[P:], rcond=None)[0]
    Af = np.column_stack([np.ones(n - P)] + [x[P - k:n - k] for k in range(1, P + 1)])
    e = x[P:] - Af @ b
    n0e = n0 - P
    pe = e[:n0e]
    z = (e - np.median(pe)) / (1.4826 * np.median(np.abs(pe - np.median(pe))) + 1e-12)
    out = {}
    for w in WS:
        if n0e < w or len(z) - n0e < w:
            continue
        t, lvl, vol = curves(z, w)
        i0 = n0e - w
        for k, D in (("lvl", np.abs(lvl)), ("vol", np.abs(vol))):
            at = D[i0]
            nb = D[max(i0 - w, 0):i0 + w + 1]
            out.update({f"ms_{k}{w}_at": at, f"ms_{k}{w}_rank": (D < at).mean(), f"ms_{k}{w}_rel": at / (D.max() + 1e-12),
                        f"ms_{k}{w}_near": nb.max(), f"ms_{k}{w}_dist": abs(int(D.argmax()) - i0) / n})
    for k in ("lvl", "vol"):
        ats = [out[f"ms_{k}{w}_at"] for w in WS if f"ms_{k}{w}_at" in out]
        rks = [out[f"ms_{k}{w}_rank"] for w in WS if f"ms_{k}{w}_rank" in out]
        out[f"ms_{k}_atmax"], out[f"ms_{k}_rankmean"] = max(ats), float(np.mean(rks))
    return out


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    items = [(str(k), g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    rows = Parallel(n_jobs=30, batch_size=64)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    G.to_parquet("outputs/factory/mosum.parquet")
    print(f"{len(G)} series, {G.shape[1]} cols in {time.time() - t0:.1f}s")
