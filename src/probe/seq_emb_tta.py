"""Like seq_emb.py but each series is embedded in several views with the OUTER ends cropped (keep the boundary region) and the
embeddings / head outputs are averaged; PCA fitted on the averaged embeddings. REVOUT: same for reversed tensors (same PCA)."""
import sys, os, numpy as np, pandas as pd, torch
CK, OUT, PFX = sys.argv[1], sys.argv[2], sys.argv[3]; sys.argv = [sys.argv[0], "0"]; import seq_pretrain as sp
from sklearn.decomposition import PCA
m = sp.Net().to(sp.dev); m.load_state_dict(torch.load(CK), strict=False); m.eval(); PRE = sp.PRE
CROPS = [(0, 0), (.1, 0), (0, .1), (.1, .1), (.2, .2)]
def view(X, fp, fq):
    X = X.clone(); msk = X[:, 2] > 0; L = msk.shape[1]; idx = torch.arange(L)
    n0 = msk[:, :PRE].sum(1); n1 = msk[:, PRE:].sum(1); lo = PRE - n0 + (fp * n0).long(); hi = PRE + n1 - (fq * n1).long()
    keep = (idx[None] >= lo[:, None]) & (idx[None] < hi[:, None]); X[:, 2] = X[:, 2] * keep; X[:, 0] = X[:, 0] * keep; X[:, 1] = X[:, 1] * keep
    return X
def embed(X):
    E, Lg = 0, 0
    for fp, fq in CROPS:
        Ei, Li = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(X), 1024):
                e = m.emb(view(X[i:i+1024].float(), fp, fq).to(sp.dev)); Ei.append(e.float().cpu()); Li.append(torch.cat([m.h(e).float(), m.aux(e).float()], 1).cpu())
        E = E + torch.cat(Ei).numpy() / len(CROPS); Lg = Lg + torch.cat(Li).numpy() / len(CROPS)
    return E, Lg
E, L = embed(torch.cat([sp.Xr, torch.tensor(np.load("outputs/arms/seq_test.npy"))])); mu, sdv = E.mean(0), E.std(0) + 1e-6
pca = PCA(32, random_state=0).fit((E - mu) / sdv); P = pca.transform((E - mu) / sdv)
ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
def frame(P, L):
    D = pd.DataFrame(P, index=pd.Index(ids, name="id"), columns=[f"{PFX}pc{i}" for i in range(32)]); D[f"{PFX}logit"] = L[:, 0]
    for i in range(1, L.shape[1]): D[f"{PFX}aux{i}"] = L[:, i]
    return D
frame(P, L).to_parquet(OUT)
from sklearn.metrics import roc_auc_score; print("zero-shot TTA", roc_auc_score(sp.yr, L[:len(sp.yr), 0]))
if os.environ.get("REVOUT"):
    Er, Lr = embed(torch.cat([torch.tensor(np.load("outputs/arms/seq_train_rev.npy")), torch.tensor(np.load("outputs/arms/seq_test_rev.npy"))])); frame(pca.transform((Er - mu) / sdv), Lr).to_parquet(os.environ["REVOUT"]); print("rev ok")
