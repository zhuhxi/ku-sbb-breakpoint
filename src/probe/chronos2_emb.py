"""Chronos-2 encoder embeddings of the pre segment (last 512 pts) and the post segment (first 512 pts), mean-pooled over patches;
features: cosine distance, L2 distance, PCA16 of (post - pre). REVSEQ=1: time-reversed series. -> OUT parquet (prefix c2e_)"""
import os, sys, time, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline
from sklearn.decomposition import PCA
OUT = sys.argv[1]; t0 = time.time(); pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda", torch_dtype=torch.bfloat16)
S = pd.concat([pd.read_parquet("data/train_series.parquet"), pd.read_parquet("data/test_series.parquet")]); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"], kind="stable")
if os.environ.get("REVSEQ"):
    n = S.groupby("id")["time"].transform("max"); S = S.assign(time=n - S["time"], period=1 - S["period"]).sort_values(["id", "time"], kind="stable")
ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
G = {k: (g["value"].to_numpy(np.float32), int((g["period"].to_numpy() == 0).sum())) for k, g in S.groupby("id", sort=False)}
def emb(ctxs):
    out = []
    for i in range(0, len(ctxs), 256):
        e, _ = pipe.embed([torch.tensor(c) for c in ctxs[i:i + 256]])
        out += [t[0, 1:-1].float().mean(0).cpu().numpy() if t.shape[1] > 2 else t[0].float().mean(0).cpu().numpy() for t in e]
    return np.stack(out)
pre = emb([G[k][0][max(0, G[k][1] - 512):G[k][1]] for k in ids]); post = emb([G[k][0][G[k][1]:G[k][1] + 512] for k in ids])
cos = 1 - (pre * post).sum(1) / (np.linalg.norm(pre, axis=1) * np.linalg.norm(post, axis=1) + 1e-9); l2 = np.linalg.norm(post - pre, axis=1)
D = pd.DataFrame(PCA(16, random_state=0).fit_transform(post - pre), index=pd.Index(ids, name="id"), columns=[f"c2e_pc{i}" for i in range(16)]); D["c2e_cos"] = cos; D["c2e_l2"] = l2
D.to_parquet(OUT); print(D.shape, f"{time.time()-t0:.0f}s")
