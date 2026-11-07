"""Harvested before/after contrast features (set H) for train and test.

    python harvest.py

Ideas borrowed from other fields / competitions, each turned into a before-vs-after contrast:
  V  variance-break tests: Inclan-Tiao ICSS and its kurtosis-robust (Kokoszka-Leipus / Sanso) form at
     the boundary vs anywhere, ARCH-LM and squared-ACF per side, Qn / Sn robust scale, scale of k-step
     changes (multi-scale), detrended residual scale;
  R  realized-volatility features (Optiver): bipower variation, jump ratio, realized quarticity,
     Hill tail index, tail-to-body quantile ratios;
  C  change-point cost (ruptures-style): Gaussian split gain at the boundary vs the best split
     anywhere, rank of the boundary among all splits, distance of the best split from the boundary;
  K  catch22 on each side, after minus before;
  Z  self-normalized contrasts: (f(after) - f(before)) / within-segment spread of f over 4 blocks.
Everything is per series, so there is no leakage between folds.
"""
import warnings

import numpy as np
import pandas as pd
import pycatch22
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)
EPS = 1e-12
LN = lambda a, b: np.log((a + EPS) / (b + EPS))


def qn_sn(x, cap=400):
    if len(x) > cap:  # strided subsample keeps the O(n^2) pairwise cost bounded
        x = x[np.linspace(0, len(x) - 1, cap).astype(int)]
    D = np.abs(x[:, None] - x[None, :])
    iu = np.triu_indices(len(x), 1)
    return np.quantile(D[iu], 0.25), np.median(np.median(D, 1))


def hill(x, frac=0.1):
    a = np.sort(np.abs(x - np.median(x)))[::-1]
    k = max(5, int(len(a) * frac))
    if a[k] <= 0:
        return np.nan
    return np.mean(np.log(a[:k] / a[k]))


def arch_lm(e, p=3):
    e2 = (e - e.mean()) ** 2
    if len(e2) < 4 * p + 10:
        return np.nan
    Y = e2[p:]
    X = np.column_stack([np.ones(len(Y))] + [e2[p - i:-i] for i in range(1, p + 1)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r = Y - X @ beta
    r2 = 1 - r.var() / (Y.var() + EPS)
    return len(Y) * r2


def acf_sq(x, lag):
    s = (x - x.mean()) ** 2
    s = s - s.mean()
    return float((s[:-lag] * s[lag:]).sum() / ((s * s).sum() + EPS))


def side_stats(x):
    """Scale / tail / volatility statistics of one segment (used raw and self-normalized)."""
    dx = np.diff(x)
    rv = np.mean(dx ** 2)
    bv = np.pi / 2 * np.mean(np.abs(dx[1:]) * np.abs(dx[:-1]))
    t = np.arange(len(x))
    res = x - np.polyval(np.polyfit(t, x, 1), t)
    q = np.quantile(x, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    out = {
        "lrv": np.log(rv + EPS), "lbv": np.log(bv + EPS), "jump": 1 - bv / (rv + EPS),
        "rq": np.mean(dx ** 4) / (rv ** 2 + EPS), "hill": hill(x), "dhill": hill(dx),
        "tail99": (q[5] - q[0]) / (q[3] - q[2] + EPS), "tail95": (q[4] - q[1]) / (q[3] - q[2] + EPS),
        "lres": np.log(res.std() + EPS), "sqacf1": acf_sq(x, 1), "sqacf2": acf_sq(x, 2),
        "dsqacf1": acf_sq(dx, 1), "arch": arch_lm(x), "darch": arch_lm(dx),
        "absacf1": np.corrcoef(np.abs(dx[1:]), np.abs(dx[:-1]))[0, 1] if dx.std() > 0 else 0.0,
    }
    for k in (1, 2, 4, 8, 16):
        if len(x) > 3 * k:
            out[f"lk{k}"] = np.log(np.std(x[k:] - x[:-k]) + EPS)
    return out


def icss(e, n0, trim=20):
    """Inclan-Tiao centred cumulative sum of squares at the boundary vs its sup over the series."""
    T = len(e)
    s = e ** 2
    C = np.cumsum(s)
    k = np.arange(1, T + 1)
    D = C / C[-1] - k / T
    it = np.sqrt(T / 2) * np.abs(D)
    om = np.sqrt(np.var(s) + EPS)  # kurtosis-robust scale of the squares (Sanso et al. kappa_1)
    kl = np.abs(C - k / T * C[-1]) / (np.sqrt(T) * om)
    mid = slice(trim, T - trim)
    return {"at": it[n0 - 1], "sup": it[mid].max(), "kl_at": kl[n0 - 1], "kl_sup": kl[mid].max(),
            "ratio": it[n0 - 1] / (it[mid].max() + EPS), "argd": (np.argmax(it[mid]) + trim - n0) / T}


def split_gain(x, n0, kind, trim=20):
    """Gaussian log-likelihood gain of splitting at k, for every k; returns boundary vs all splits."""
    T = len(x)
    c1, c2 = np.cumsum(x), np.cumsum(x ** 2)
    k = np.arange(trim, T - trim)
    nl, nr = k, T - k
    if kind == "meanvar":
        vl = c2[k - 1] / nl - (c1[k - 1] / nl) ** 2
        vr = (c2[-1] - c2[k - 1]) / nr - ((c1[-1] - c1[k - 1]) / nr) ** 2
        vall = x.var()
    else:  # "var": zero-mean variance change (used on differences)
        vl, vr, vall = c2[k - 1] / nl, (c2[-1] - c2[k - 1]) / nr, np.mean(x ** 2)
    g = T * np.log(vall + EPS) - nl * np.log(vl + EPS) - nr * np.log(vr + EPS)
    j = np.clip(n0 - trim, 0, len(k) - 1)
    return {"at": g[j] / T, "max": g.max() / T, "ratio": g[j] / (g.max() + EPS),
            "rank": np.mean(g > g[j]), "argd": abs(k[np.argmax(g)] - n0) / T}


def catch22(x):
    r = pycatch22.catch22_all(list(x), catch24=False)
    return dict(zip(r["names"], r["values"]))


def harvest(sid, v, p):
    x0, x1 = v[p == 0], v[p == 1]
    x = np.concatenate([x0, x1])
    n0, f = len(x0), {"id": sid}
    if min(len(x0), len(x1)) < 40:
        return f
    dx = np.diff(x)

    # V: variance-break tests on residuals from per-segment medians and on differences
    e = np.concatenate([x0 - np.median(x0), x1 - np.median(x1)])
    for tag, z, b in (("e", e, n0), ("d", dx - np.median(dx), n0 - 1)):
        for k, val in icss(z, b).items():
            f[f"H_icss_{tag}_{k}"] = val
    for tag, a, b in (("x", x0, x1), ("d", np.diff(x0), np.diff(x1))):
        qa, sa = qn_sn(a)
        qb, sb = qn_sn(b)
        f[f"H_lqn_{tag}"], f[f"H_lsn_{tag}"] = LN(qb, qa), LN(sb, sa)

    # R + V per-side stats: after-minus-before, plus self-normalized version
    s0, s1 = side_stats(x0), side_stats(x1)
    for k in s0:
        f[f"H_{k}_d"] = s1.get(k, np.nan) - s0[k]
    blocks = [side_stats(b) for b in np.array_split(x0, 4) + np.array_split(x1, 4)]
    for k in ("lrv", "lbv", "jump", "rq", "lres", "sqacf1", "absacf1", "lk1", "lk4", "tail95", "hill"):
        bv = np.array([b.get(k, np.nan) for b in blocks], float)
        spread = np.sqrt((np.nanvar(bv[:4]) + np.nanvar(bv[4:])) / 2) / 2  # block sd -> segment sd (4 blocks)
        f[f"H_z_{k}"] = (s1.get(k, np.nan) - s0[k]) / (spread + 1e-6)
    f["H_lk_profile"] = (s1.get("lk16", np.nan) - s1["lk1"]) - (s0.get("lk16", np.nan) - s0["lk1"])

    # C: change-point cost, is the boundary the best split?
    for tag, z, b, kind in (("mv", x, n0, "meanvar"), ("dv", dx, n0 - 1, "var"),
                            ("rv", e, n0, "var")):
        for k, val in split_gain(z, b, kind).items():
            f[f"H_cp_{tag}_{k}"] = val

    # K: catch22 contrast
    c0, c1 = catch22(x0), catch22(x1)
    for i, k in enumerate(c0):
        f[f"H_c22_{i:02d}_d"] = c1[k] - c0[k]
    return f


def build(path):
    df = pd.read_parquet(path).sort_values(["id", "time"], kind="stable")
    groups = [(sid, g["value"].to_numpy(float), g["period"].to_numpy(int)) for sid, g in df.groupby("id", sort=False)]
    rows = Parallel(n_jobs=28, batch_size=64)(delayed(harvest)(*g) for g in groups)
    return pd.DataFrame(rows).set_index("id").replace([np.inf, -np.inf], np.nan).astype(np.float32)


if __name__ == "__main__":
    for split in ("train", "test"):
        H = build(f"data/{split}_series.parquet")
        H.to_parquet(f"outputs/harvest_{split}.parquet")
        print(split, H.shape, "NaN frac", round(float(H.isna().mean().mean()), 4), flush=True)
