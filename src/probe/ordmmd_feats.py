"""Scale-free distribution-change features pre vs post: ordinal-pattern (order 3, 4) JS divergence + permutation-entropy change,
on raw and AR(5)-pre residuals; RBF MMD^2 of 3-dim delay embeddings (pre-standardised), raw and |resid|. -> outputs/factory/ordmmd.parquet"""
import numpy as np, pandas as pd
from math import factorial
rng = np.random.default_rng(0)
def ordp(x, m):
    W = np.lib.stride_tricks.sliding_window_view(x, m); k = np.argsort(W, 1); c = (k * (m ** np.arange(m))).sum(1)
    return np.bincount(np.unique(c, return_inverse=True)[1] if False else c, minlength=m ** m)
def js(p, q):
    p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2; f = lambda a: np.sum(a[a > 0] * np.log(a[a > 0] / m[a > 0])); return (f(p) + f(q)) / 2
def H(p): p = p / p.sum(); p = p[p > 0]; return -(p * np.log(p)).sum()
def mmd(A, B):
    A = A[rng.choice(len(A), min(len(A), 150), replace=False)]; B = B[rng.choice(len(B), min(len(B), 150), replace=False)]
    Z = np.vstack([A, B]); D = ((Z[:, None] - Z[None]) ** 2).sum(-1); g = 1 / (np.median(D[D > 0]) + 1e-9); K = np.exp(-g * D); n = len(A)
    return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()
def one(v, n0):
    d = {}; x0 = v[:n0]; z = (v - x0.mean()) / (x0.std() + 1e-9); q = 5
    Zl = np.column_stack([np.ones(len(z) - q)] + [z[q - j - 1:len(z) - j - 1] for j in range(q)])
    c = np.linalg.lstsq(Zl[:n0 - q], z[q:n0], rcond=None)[0]; e = np.r_[np.zeros(q), z[q:] - Zl @ c]
    for nm, s in (("raw", z), ("res", e)):
        a, b = s[q:n0], s[n0:]
        for m in (3, 4):
            pa, pb = ordp(a, m) + .5, ordp(b, m) + .5; d[f"op_js{m}_{nm}"] = js(pa, pb); d[f"op_dH{m}_{nm}"] = H(pb) - H(pa)
    emb = lambda s: np.lib.stride_tricks.sliding_window_view(s, 3)
    d["mmd_raw"] = mmd(emb(z[:n0]), emb(z[n0:])); d["mmd_absres"] = mmd(emb(np.abs(e[q:n0])), emb(np.abs(e[n0:])))
    return d
rows = {}
for p in ("data/train_series.parquet", "data/test_series.parquet"):
    S = pd.read_parquet(p); S["id"] = S.id.astype(str); S = S.sort_values(["id", "time"], kind="stable")
    for i, g in S.groupby("id", sort=False): rows[i] = one(g.value.to_numpy(float), int((g.period.to_numpy() == 0).sum()))
D = pd.DataFrame.from_dict(rows, orient="index"); D.index.name = "id"; D.to_parquet("outputs/factory/ordmmd.parquet"); print(D.shape)
