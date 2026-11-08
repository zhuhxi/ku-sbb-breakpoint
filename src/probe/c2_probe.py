"""Chronos-2 encoder as a backbone. Windows: last W pre points + first W post points (NaN-padded), boundary fixed at position W.
Frozen encoder -> tokens; pooled pre / post / diff / extra tokens -> OUT npy (float16). Token layout checked by a perturbation test."""
import sys, os, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline
W = 256; dev = "cuda"
p = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=dev, torch_dtype=torch.float32); m = p.model.eval()
def windows(path_list, rev=False):
    S = pd.concat([pd.read_parquet(f) for f in path_list]); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"], kind="stable")
    if rev:
        n = S.groupby("id")["time"].transform("max"); S = S.assign(time=n - S["time"], period=1 - S["period"]).sort_values(["id", "time"], kind="stable")
    ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
    G = {k: (g["value"].to_numpy(np.float32), int((g["period"].to_numpy() == 0).sum())) for k, g in S.groupby("id", sort=False)}
    X = np.full((len(ids), 2 * W), np.nan, np.float32)
    for i, k in enumerate(ids):
        v, n0 = G[k]; a = v[max(0, n0 - W):n0]; b = v[n0:n0 + W]; X[i, W - len(a):W] = a; X[i, W:W + len(b)] = b
    return X
if __name__ == "__main__":
    if sys.argv[1] == "layout":
        x = torch.randn(1, 2 * W); e0 = m.encode(context=x.to(dev), group_ids=torch.zeros(1, dtype=torch.long, device=dev))[0][0]
        x2 = x.clone(); x2[0, -16:] += 5; e1 = m.encode(context=x2.to(dev), group_ids=torch.zeros(1, dtype=torch.long, device=dev))[0][0]
        x3 = x.clone(); x3[0, :16] += 5; e2 = m.encode(context=x3.to(dev), group_ids=torch.zeros(1, dtype=torch.long, device=dev))[0][0]
        d1 = (e1 - e0).norm(dim=-1).detach().cpu().numpy(); d2 = (e2 - e0).norm(dim=-1).detach().cpu().numpy()
        print("change last 16 pts -> token norms", np.round(d1, 1)); print("change first 16 pts ->", np.round(d2, 1)); sys.exit()
    rev = sys.argv[1] == "rev"; X = windows(["data/train_series.parquet", "data/test_series.parquet"], rev); F = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i:i + 512], device=dev); e = m.encode(context=xb, group_ids=torch.arange(len(xb), device=dev))[0][0].float()
            pre, post = e[:, :W // 16].mean(1), e[:, W // 16:2 * W // 16].mean(1)
            F.append(torch.cat([pre, post, post - pre, e[:, -2], e[:, -1]], 1).cpu().numpy().astype(np.float16))
    np.save(f"outputs/arms/c2tok_{'rev' if rev else 'fwd'}.npy", np.concatenate(F)); print("saved", np.concatenate(F).shape)
