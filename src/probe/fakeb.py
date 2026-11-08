"""Self-normalised CNN score: zero-shot logit of a nuis-pretrained CNN at the TRUE boundary with the post segment cut to D points,
and at FAKE boundaries inside the pre segment (pre = v[:n0-D-s], post = v[n0-D-s:n0-s], s in SHIFTS). Features: true, fake mean/max,
true - fake mean. -> OUT parquet (prefix fb_). Tensor layout as probe/seq_prep.py."""
import os, sys, numpy as np, pandas as pd, torch
CK, OUT, D = sys.argv[1], sys.argv[2], int(sys.argv[3]); sys.argv = [sys.argv[0], "0"]; sys.path.insert(0, "probe"); import seq_pretrain as sp
from joblib import Parallel, delayed
PRE, POST, q = 960, 736, 10; SHIFTS = (0, D // 2, D)
def tens(v, n0):
    out = np.zeros((3, PRE + POST), np.float16)
    if n0 < 3 * q or n0 > PRE or len(v) - n0 > POST or len(v) - n0 < 4: return None
    x0 = v[:n0]; med = np.median(x0); mad = 1.4826 * np.median(np.abs(x0 - med)) + 1e-8; z = (v - med) / mad
    Zl = np.column_stack([np.ones(len(z) - q)] + [z[q - j - 1:len(z) - j - 1] for j in range(q)])
    c = np.linalg.lstsq(Zl[:n0 - q], z[q:n0], rcond=None)[0]; e = np.r_[np.zeros(q), z[q:] - Zl @ c]; e /= e[q:n0].std() + 1e-8
    lo = PRE - n0; out[0, lo:PRE + len(v) - n0] = np.clip(z, -30, 30); out[1, lo:PRE + len(v) - n0] = np.clip(e, -30, 30); out[2, lo:PRE + len(v) - n0] = 1; out[1, lo:lo + q] = 0
    return out
def one(v, n0):
    T = [tens(v[:n0 + D], n0)] + [tens(v[:n0 - s], n0 - D - s) for s in SHIFTS]
    return [t if t is not None else np.zeros((3, PRE + POST), np.float16) for t in T], [t is not None for t in T]
S = pd.concat([pd.read_parquet("data/train_series.parquet"), pd.read_parquet("data/test_series.parquet")]); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"], kind="stable")
if os.environ.get("REVSEQ"):
    n = S.groupby("id")["time"].transform("max"); S = S.assign(time=n - S["time"], period=1 - S["period"]).sort_values(["id", "time"], kind="stable")
ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
G = {k: (g["value"].to_numpy(float), int((g["period"].to_numpy() == 0).sum())) for k, g in S.groupby("id", sort=False)}
R = Parallel(n_jobs=30, batch_size=256)(delayed(one)(*G[k]) for k in ids)
m = sp.Net().to(sp.dev); m.load_state_dict(torch.load(CK), strict=False)
L = np.full((len(ids), 1 + len(SHIFTS)), np.nan)
for j in range(1 + len(SHIFTS)):
    X = torch.tensor(np.stack([r[0][j] for r in R])); ok = np.array([r[1][j] for r in R]); p = sp.predict(m, X); L[ok, j] = p[ok]
F = pd.DataFrame(index=pd.Index(ids, name="id")); F["fb_true"] = L[:, 0]; F["fb_fmean"] = np.nanmean(L[:, 1:], 1); F["fb_fmax"] = np.nanmax(L[:, 1:], 1)
F["fb_delta"] = F.fb_true - F.fb_fmean; F["fb_dmax"] = F.fb_true - F.fb_fmax; F.columns = [c.replace("fb_", f"fb{D}_") for c in F.columns]; F.to_parquet(OUT); print(F.shape, F.notna().mean().round(2).to_dict())
