"""ADIA 2025 winner (Alphabot) ideas not yet in our feature set (research/README.md):
  band occupancy : share of points within m0 +- c*s0 (pre-part mean / sd as the ruler), after minus before, c in .1 .25 .5 1 2;
                   also locally (last 100 before vs first 50 after the boundary) for c in .5 1 2
  decile JSD / Hellinger : the after part binned on the pre-part deciles (same distribution -> uniform 10%) vs the pre part
  slab JSD : the same on the middle 40-60% slab of the pre part, 5 sub-bins (shape of the centre only)
on the raw series and on AR(10) residuals (factory.py RES10).

    python probe/band_feats.py     # train + test -> outputs/factory/band.parquet
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()
src = open("factory.py").read()
exec(src[src.index("def _ar_resid"):src.index("def _devol")])
C = (.1, .25, .5, 1, 2)
EPS = 1e-6


def jsd(p, q):
    p, q = p + EPS, q + EPS
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return .5 * np.sum(p * np.log(p / m)) + .5 * np.sum(q * np.log(q / m))


def hell(p, q):
    return np.sqrt(.5 * np.sum((np.sqrt(p / p.sum()) - np.sqrt(q / q.sum())) ** 2))


def fam(v, n0, tag):
    a, b = v[:n0], v[n0:]
    m0, s0 = a.mean(), a.std() + 1e-12
    r = {}
    for c in C:
        r[f"bd_{tag}_occ{c}"] = np.mean(np.abs(b - m0) <= c * s0) - np.mean(np.abs(a - m0) <= c * s0)
    aw, bw = a[-100:], b[:50]
    for c in (.5, 1, 2):
        r[f"bd_{tag}_loc{c}"] = np.mean(np.abs(bw - m0) <= c * s0) - np.mean(np.abs(aw - m0) <= c * s0)
    e = np.quantile(a, np.linspace(0, 1, 11)[1:-1])
    pa = np.bincount(np.searchsorted(e, a), minlength=10).astype(float)
    pb = np.bincount(np.searchsorted(e, b), minlength=10).astype(float)
    r[f"bd_{tag}_jsd10"], r[f"bd_{tag}_hel10"] = jsd(pa, pb), hell(pa, pb)
    lo, hi = np.quantile(a, [.4, .6])
    sa, sb = a[(a >= lo) & (a <= hi)], b[(b >= lo) & (b <= hi)]
    if len(sb) >= 5:
        e5 = np.linspace(lo, hi, 6)[1:-1]
        qa = np.bincount(np.searchsorted(e5, sa), minlength=5).astype(float)
        qb = np.bincount(np.searchsorted(e5, sb), minlength=5).astype(float)
        r[f"bd_{tag}_slab"] = jsd(qa, qb)
    else:
        r[f"bd_{tag}_slab"] = np.nan
    return r


def one(i, v, n0):
    return dict(id=i, **fam(v, n0, "raw"), **fam(_ar_resid(v, 10), n0, "res"))


S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
ids = S.id.to_numpy()
cut = np.flatnonzero(ids[1:] != ids[:-1]) + 1
N0 = S[S.period == 0].groupby("id").size()
G = pd.DataFrame(Parallel(n_jobs=30, batch_size=128)(delayed(one)(i, x, N0[i]) for i, x in zip(ids[np.r_[0, cut]], np.split(S.value.to_numpy(), cut)))).set_index("id")
G.index = G.index.astype(str)
G.to_parquet("outputs/factory/band.parquet")
print(G.shape, f"{time.time() - t0:.1f}s")
