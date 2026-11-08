"""Label-free features from the synthetic-pretrained CNN: 512-d pooled embedding + zero-shot logit for real train+test,
PCA(32) fitted on train+test (unsupervised) -> outputs/factory/seqemb.parquet"""
import sys, numpy as np, pandas as pd, torch
CK, OUT, PFX = sys.argv[1], sys.argv[2], sys.argv[3]; sys.argv = [sys.argv[0], "0"]; import seq_pretrain as sp
from sklearn.decomposition import PCA
m = sp.Net().to(sp.dev); m.load_state_dict(torch.load(CK), strict=False); m.eval()
import os
def embed(X):
    E, L = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 1024):
            e = m.emb(X[i:i+1024].to(sp.dev)); E.append(e.float().cpu()); L.append(torch.cat([m.h(e).float(), m.aux(e).float()], 1).cpu())
    return torch.cat(E).numpy(), torch.cat(L).numpy()
E, L = embed(torch.cat([sp.Xr, torch.tensor(np.load("outputs/arms/seq_test.npy"))])); mu, sdv = E.mean(0), E.std(0) + 1e-6
NPC = int(os.environ.get("NPC", 32)); pca = PCA(NPC, random_state=0).fit((E - mu) / sdv); P = pca.transform((E - mu) / sdv)
REVOUT = os.environ.get("REVOUT")   # also embed the time-reversed tensors with the SAME normalisation / PCA
if REVOUT:
    Er, Lr = embed(torch.cat([torch.tensor(np.load("outputs/arms/seq_train_rev.npy")), torch.tensor(np.load("outputs/arms/seq_test_rev.npy"))])); Pr = pca.transform((Er - mu) / sdv)
ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
D = pd.DataFrame(P, index=pd.Index(ids, name="id"), columns=[f"{PFX}pc{i}" for i in range(NPC)]); D[f"{PFX}logit"] = L[:, 0]
if "_mt" in CK:
    for i in range(1, L.shape[1]): D[f"{PFX}aux{i}"] = L[:, i]
D.to_parquet(OUT); print(D.shape)
if REVOUT:
    Dr = pd.DataFrame(Pr, index=D.index, columns=D.columns[:NPC]); Dr[f"{PFX}logit"] = Lr[:, 0]
    if "_mt" in CK:
        for i in range(1, Lr.shape[1]): Dr[f"{PFX}aux{i}"] = Lr[:, i]
    Dr.to_parquet(REVOUT); print("rev", Dr.shape)
