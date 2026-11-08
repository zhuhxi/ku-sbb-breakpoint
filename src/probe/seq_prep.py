"""Boundary-aligned sequence tensors for NN arms: pre part right-aligned to index PRE, post part from PRE on.
channels: z (pre median/MAD), AR(10)-pre residual / pre resid sd, mask. -> outputs/arms/seq_{train,test}.npy (float16)"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "probe"); import feats
PRE, POST = 960, 736
def prep(path, ids):
    S = pd.read_parquet(path); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"], kind="stable")
    if __import__("os").environ.get("REVSEQ"):   # time-reversed series (boundary at n - n0)
        n = S.groupby("id")["time"].transform("max"); S = S.assign(time=n - S["time"], period=1 - S["period"]).sort_values(["id", "time"], kind="stable")
    out = np.zeros((len(ids), 3, PRE + POST), np.float16); pos = {k: i for i, k in enumerate(ids)}
    for sid, g in S.groupby("id", sort=False):
        v = g["value"].to_numpy(float); p = g["period"].to_numpy(); n0 = int((p == 0).sum())
        x0 = v[:n0]; med = np.median(x0); mad = 1.4826 * np.median(np.abs(x0 - med)) + 1e-8; z = (v - med) / mad
        q = 10; Zl = np.column_stack([np.ones(len(z) - q)] + [z[q - j - 1:len(z) - j - 1] for j in range(q)])
        coef = np.linalg.lstsq(Zl[:n0 - q], z[q:n0], rcond=None)[0]; e = np.r_[np.zeros(q), z[q:] - Zl @ coef]
        e /= e[q:n0].std() + 1e-8
        i = pos[sid]; lo = PRE - n0
        out[i, 0, lo:PRE + len(v) - n0] = np.clip(z, -30, 30); out[i, 1, lo:PRE + len(v) - n0] = np.clip(e, -30, 30)
        out[i, 2, lo:PRE + len(v) - n0] = 1; out[i, 1, lo:lo + q] = 0
    return out
SFX = "_rev" if __import__("os").environ.get("REVSEQ") else ""
np.save(f"outputs/arms/seq_train{SFX}.npy", prep("data/train_series.parquet", list(feats.lab.index)))
np.save(f"outputs/arms/seq_test{SFX}.npy", prep("data/test_series.parquet", list(feats.sub["id"])))
print("ok")
