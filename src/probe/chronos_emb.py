"""Chronos-Bolt encoder embeddings of the pre segment (last 512) and the post segment (first 512), mean-pooled; features = cosine and
L2 distance between them + PCA16 of (post - pre). Label-free. -> outputs/factory/chronos_emb.parquet"""
import sys, time, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline
from sklearn.decomposition import PCA
t0 = time.time(); pipe = BaseChronosPipeline.from_pretrained(sys.argv[1], device_map="cuda", torch_dtype=torch.float32)
S = pd.concat([pd.read_parquet("data/train_series.parquet"), pd.read_parquet("data/test_series.parquet")]); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"], kind="stable")
ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
G = {k: (g["value"].to_numpy(np.float32), int((g["period"].to_numpy() == 0).sum())) for k, g in S.groupby("id", sort=False)}
def emb(segs):
    out = []
    for i in range(0, len(segs), 256):
        e, _ = pipe.embed([torch.tensor(s) for s in segs[i:i + 256]]); out.append(e.float().mean(1).cpu().numpy())
    return np.concatenate(out)
Ep = emb([G[k][0][max(0, G[k][1] - 512):G[k][1]] for k in ids]); Eq = emb([G[k][0][G[k][1]:G[k][1] + 512] for k in ids])
cos = (Ep * Eq).sum(1) / (np.linalg.norm(Ep, axis=1) * np.linalg.norm(Eq, axis=1) + 1e-9); l2 = np.linalg.norm(Ep - Eq, axis=1) / (np.linalg.norm(Ep, axis=1) + 1e-9)
P = PCA(16, random_state=0).fit_transform(Eq - Ep)
D = pd.DataFrame(P, index=pd.Index(ids, name="id"), columns=[f"cb_ed{i}" for i in range(16)]); D["cb_cos"] = cos; D["cb_l2"] = l2
D.to_parquet("outputs/factory/chronos_emb.parquet"); print(D.shape, Ep.shape, f"{time.time()-t0:.0f}s")
from sklearn.metrics import roc_auc_score
y = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"].reindex(D.index[:24000]); T = D.iloc[:24000]
print({c: round(max(a := roc_auc_score(y, T[c]), 1 - a), 3) for c in ["cb_cos", "cb_l2", "cb_ed0", "cb_ed1"]})
