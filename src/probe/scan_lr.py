"""Joint-LR change-point scan: the g_jlr test (probe/joint_lr.py) evaluated at EVERY split tau, from cumulative
cross-products (no per-tau refit loop). Rows t >= 2, w_t = [1, t/n, x_{t-1}, x_{t-2}, y=x_t]:
  null : y ~ 1 + t/n + x_{t-1} + x_{t-2}, one sd
  alt  : null + s_tau + s_tau x_{t-1} (s_tau = 1{t >= tau}), sd per side (residuals of the OLS alt fit)
LR(tau) = loglik(alt) - loglik(null). Features (tau in [M, n - M]):
  sl_b     LR(n0)                       sl_rel   LR(n0) / max LR
  sl_gap   LR(n0) - median LR           sl_rank  share of tau with LR < LR(n0)
  sl_late  max LR on [n0, n0 + 20] - LR(n0)   (late jumps, ~10% of breaks)
  sl_early max LR on [n0 - 20, n0] - LR(n0)   sl_dist |argmax - n0| / n     sl_max  max LR

    python probe/scan_lr.py     # train + test -> outputs/factory/scan_lr.parquet
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

M = 20


def one(v, n0):
    n = len(v)
    t = np.arange(2, n)
    W = np.column_stack([np.ones(n - 2), t / n, v[1:-1], v[:-2], v[2:]])
    O = W[:, :, None] * W[:, None, :]                         # (T, 5, 5)
    Gp = np.cumsum(O[::-1], 0)[::-1]                          # Gp[i] = sum over rows >= i
    G = Gp[0]
    Z, y = slice(0, 4), 4
    ZZ, Zy, yy = G[Z, Z], G[Z, y], G[y, y]
    b0 = np.linalg.solve(ZZ, Zy)
    l0 = -0.5 * len(t) * np.log((yy - b0 @ Zy) / len(t) + 1e-12)
    taus = np.arange(M, n - M)
    P = Gp[taus - 2]                                          # post sums for each tau (row index = tau - 2)
    U = [0, 2]                                                # step regressors: s*1, s*x_{t-1}
    k = len(taus)
    XX = np.zeros((k, 6, 6))
    XX[:, :4, :4] = ZZ
    XX[:, :4, 4:] = P[:, :4][:, :, U]
    XX[:, 4:, :4] = P[:, U][:, :, :4]
    XX[:, 4:, 4:] = P[:, U][:, :, U]
    Xy = np.concatenate([np.broadcast_to(Zy, (k, 4)), P[:, U, y]], 1)
    beta = np.linalg.solve(XX + 1e-9 * np.eye(6), Xy[:, :, None])[:, :, 0]
    # post-side design rows are [Z, Z_U]: X'X restricted to post rows, X'y post, y'y post
    Pz = P[:, :4, :4]
    XXp = np.zeros((k, 6, 6))
    XXp[:, :4, :4] = Pz
    XXp[:, :4, 4:] = Pz[:, :, U]
    XXp[:, 4:, :4] = Pz[:, U, :]
    XXp[:, 4:, 4:] = Pz[:, U][:, :, U]
    Xyp = np.concatenate([P[:, :4, y], P[:, U, y]], 1)
    rss = yy - np.einsum("ki,ki->k", beta, Xy)
    rp = P[:, y, y] - 2 * np.einsum("ki,ki->k", beta, Xyp) + np.einsum("ki,kij,kj->k", beta, XXp, beta)
    ra = rss - rp
    nb = len(t) - (taus - 2)
    na = len(t) - nb
    LR = -0.5 * (na * np.log(np.maximum(ra, 1e-12) / na) + nb * np.log(np.maximum(rp, 1e-12) / nb)) - l0
    i0 = int(np.clip(n0 - M, 0, k - 1))
    L0 = LR[i0]
    return {"sl_b": L0, "sl_max": LR.max(), "sl_rel": L0 / (LR.max() + 1e-9), "sl_gap": L0 - np.median(LR),
            "sl_rank": (LR < L0).mean(), "sl_late": LR[i0:i0 + 21].max() - L0,
            "sl_early": LR[max(i0 - 20, 0):i0 + 1].max() - L0, "sl_dist": abs(taus[LR.argmax()] - n0) / n}


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    items = [(str(k), g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    t1 = time.time()
    rows = Parallel(n_jobs=30, batch_size=64)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    G.to_parquet("outputs/factory/scan_lr.parquet")
    print(f"{len(G)} series, compute {time.time() - t1:.1f}s, total {time.time() - t0:.1f}s")
