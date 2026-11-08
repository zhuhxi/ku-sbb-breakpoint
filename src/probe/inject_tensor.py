"""Self-supervised data on REAL series (train+test, labels never used): for each series and variant v, inject a synthetic lasting
break after the boundary (level shift in pre-sd units, or log-variance change, or both) -> aligned tensors like seq_prep.py.
    python probe/inject_tensor.py V OUT.npz SEED   (saves X of injected copies; originals are seq_train/test.npy)"""
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, "probe"); import feats
V, OUT, SEED = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]); PRE, POST, q = 960, 736, 10
def tensor(v, n0):
    out = np.zeros((3, PRE + POST), np.float16); x0 = v[:n0]; med = np.median(x0); mad = 1.4826 * np.median(np.abs(x0 - med)) + 1e-8; z = (v - med) / mad
    Zl = np.column_stack([np.ones(len(z) - q)] + [z[q - j - 1:len(z) - j - 1] for j in range(q)])
    c = np.linalg.lstsq(Zl[:n0 - q], z[q:n0], rcond=None)[0]; e = np.r_[np.zeros(q), z[q:] - Zl @ c]; e /= e[q:n0].std() + 1e-8
    lo = PRE - n0; out[0, lo:PRE + len(v) - n0] = np.clip(z, -30, 30); out[1, lo:PRE + len(v) - n0] = np.clip(e, -30, 30); out[2, lo:PRE + len(v) - n0] = 1; out[1, lo:lo + q] = 0
    return out
def inject(v, n0, rs):
    v = v.copy(); sd = v[:n0].std() + 1e-9; mu = v[n0:].mean(); kind = rs.integers(3)
    if kind in (0, 2): v[n0:] += rs.choice([-1, 1]) * rs.uniform(.15, 1.5) * sd
    if kind in (1, 2): k = np.exp(rs.choice([-1, 1]) * rs.uniform(.1, .7)); v[n0:] = mu + (v[n0:] - mu) * k
    return v
def one(v, n0, s): return tensor(inject(v, n0, np.random.default_rng(SEED * 100000 + s)), n0)
ids = list(feats.lab.index) + list(feats.sub["id"]); S = pd.concat([pd.read_parquet("data/train_series.parquet"), pd.read_parquet("data/test_series.parquet")])
S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"], kind="stable"); G = {k: (g["value"].to_numpy(float), int((g["period"].to_numpy() == 0).sum())) for k, g in S.groupby("id", sort=False)}
X = np.stack(Parallel(n_jobs=30, batch_size=256)(delayed(one)(*G[k], i + V * len(ids)) for i, k in enumerate(ids))); np.savez(OUT, X=X); print(OUT, X.shape)
