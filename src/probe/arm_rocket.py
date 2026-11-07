"""MiniRocket-style random dilated kernels on boundary-aligned (z, AR10 residual) channels; PPV and mean pooled separately
pre / post / first 40 post / last 40 pre -> differences -> features parquet outputs/factory/rocket.parquet (train+test ids).
    CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/deeplearning/bin/python probe/arm_rocket.py [n_kernels]"""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, "probe"); import feats
K = int(sys.argv[1]) if len(sys.argv) > 1 else 400; PRE = 960; dev = "cuda"
S = torch.tensor(np.vstack([np.load("outputs/arms/seq_train.npy"), np.load("outputs/arms/seq_test.npy")]).astype(np.float32))
g = torch.Generator().manual_seed(0); rng = np.random.default_rng(0)
W = torch.tensor(rng.choice([-1., 2.], size=(K, 9), p=[2/3, 1/3]), dtype=torch.float32); W -= W.mean(1, keepdim=True)
dil = rng.choice([1, 2, 4, 8, 16, 32], K); ch = rng.integers(0, 2, K)
out = []
with torch.no_grad():
    # bias from quantiles of the response on a sample of pre parts
    for i0 in range(0, len(S), 800):
        x = S[i0:i0+800].to(dev); m = x[:, 2]; rows = []
        for k in range(K):
            d = int(dil[k]); r = F.conv1d(x[:, ch[k]:ch[k]+1], W[k].view(1, 1, 9).to(dev), padding=4 * d, dilation=d)[:, 0]
            if i0 == 0 and k == 0: pass
            rows.append(r)
        R = torch.stack(rows, 1)  # n, K, T
        if i0 == 0:
            mp = m[:, :PRE].bool(); smp = R[:200, :, :PRE].permute(1, 0, 2)[:, mp[:200]]
            B = torch.quantile(smp[:, ::7], torch.tensor(rng.uniform(.1, .9, K), dtype=torch.float32, device=dev), dim=1).diagonal()
        P = (R > B[None, :, None]).float() * m[:, None]; A = R.abs() * m[:, None]
        def pool(Z, lo, hi): w = m[:, lo:hi]; return (Z[..., lo:hi]).sum(-1) / (w.sum(-1, keepdim=True) + 1e-6)
        pp, pq, pn, pl = pool(P, 0, PRE), pool(P, PRE, None), pool(P, PRE, PRE + 40), pool(P, PRE - 40, PRE)
        ap, aq = pool(A, 0, PRE), pool(A, PRE, None)
        out.append(torch.cat([pq - pp, pn - pl, torch.log((aq + 1e-3) / (ap + 1e-3))], 1).cpu().numpy())
        print(i0, flush=True)
O = np.vstack(out).astype(np.float32)
ids = list(feats.lab.index) + list(feats.sub["id"])
cols = [f"rk_ppvd_{k}" for k in range(K)] + [f"rk_near_{k}" for k in range(K)] + [f"rk_lam_{k}" for k in range(K)]
pd.DataFrame(O, index=pd.Index(ids, name="id"), columns=cols).to_parquet("outputs/factory/rocket.parquet"); print(O.shape)
