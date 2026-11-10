"""LOF break features: fit LocalOutlierFactor(novelty) on pre-boundary delay-embedded points, score post points (and the reverse).

    python probe/lof_feat.py OUT.parquet [m=3] [k=20]
"""
import os, sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import LocalOutlierFactor

OUT = sys.argv[1]; M = int(sys.argv[2]) if len(sys.argv) > 2 else 3; K = int(sys.argv[3]) if len(sys.argv) > 3 else 20
CAP = 400


def emb(x, m):
    return np.lib.stride_tricks.sliding_window_view(x, m) if m > 1 else x[:, None]


def side(A, B, k):
    """fit on A, score B; returns outlier factors of B (>1 = outlier) and in-sample factors of A."""
    rng = np.random.default_rng(0)
    if len(A) > CAP: A = A[rng.choice(len(A), CAP, replace=False)]
    if len(B) > CAP: B = B[rng.choice(len(B), CAP, replace=False)]
    l = LocalOutlierFactor(n_neighbors=min(k, len(A) - 1), novelty=True).fit(A)
    return -l.score_samples(B), -l.negative_outlier_factor_


def one(sid, v, p):
    pre, post = v[p == 0], v[p == 1]
    mu, sd = pre.mean(), pre.std() + 1e-9
    f = {"id": sid}
    for nm, a, b in (("v", (pre - mu) / sd, (post - mu) / sd), ("d", np.diff(pre) / (np.diff(pre).std() + 1e-9), np.diff(post) / (np.diff(pre).std() + 1e-9))):
        for m in (1, M):
            A, B = emb(a, m), emb(b, m)
            if len(A) < 30 or len(B) < 30: continue
            sb, sa = side(A, B, K); ra, rb = side(B, A, K)   # forward: post vs pre model; reverse: pre vs post model
            t = f"lof_{nm}{m}"
            f[f"{t}_post_mean"] = np.log(sb).mean() - np.log(sa).mean()
            f[f"{t}_post_q90"] = np.quantile(sb, .9) / np.quantile(sa, .9)
            f[f"{t}_post_frac"] = (sb > np.quantile(sa, .95)).mean()
            f[f"{t}_rev_mean"] = np.log(ra).mean() - np.log(rb).mean()
            f[f"{t}_sym"] = f[f"{t}_post_mean"] + f[f"{t}_rev_mean"]
    return f


rows = []
for fn in ("data/train_series.parquet", "data/test_series.parquet"):
    d = pd.read_parquet(fn)
    g = [(sid, x.value.to_numpy(np.float64), x.period.to_numpy()) for sid, x in d.groupby("id", sort=False)]
    if os.environ.get("REVSER"): g = [(s, v[::-1].copy(), 1 - p[::-1]) for s, v, p in g]   # time-reversed series: post becomes pre
    rows += Parallel(n_jobs=24, batch_size=64)(delayed(one)(*a) for a in g)
    print(fn, len(g), flush=True)
D = pd.DataFrame(rows).set_index("id"); D.to_parquet(OUT); print(D.shape, D.isna().mean().max())
