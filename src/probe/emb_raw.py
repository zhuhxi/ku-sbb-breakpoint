"""Save the raw 512-d pooled embeddings of a pretrained CNN for real train+test (and reversed) -> outputs/arms/embraw_{TAG}.npy"""
import sys, os, numpy as np, torch
CK, TAG = sys.argv[1], sys.argv[2]; sys.argv = [sys.argv[0], "0"]; import seq_pretrain as sp
m = sp.Net().to(sp.dev); m.load_state_dict(torch.load(CK), strict=False); m.eval()
def embed(X):
    E = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 1024): E.append(m.emb(X[i:i+1024].to(sp.dev)).float().cpu())
    return torch.cat(E).numpy().astype(np.float32)
np.save(f"outputs/arms/embraw_{TAG}.npy", embed(torch.cat([sp.Xr, torch.tensor(np.load("outputs/arms/seq_test.npy"))])))
np.save(f"outputs/arms/embraw_{TAG}_rev.npy", embed(torch.cat([torch.tensor(np.load("outputs/arms/seq_train_rev.npy")), torch.tensor(np.load("outputs/arms/seq_test_rev.npy"))]))); print("ok")
