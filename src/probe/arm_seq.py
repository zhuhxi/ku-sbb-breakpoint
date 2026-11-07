"""Hybrid NN arm: dilated 1D CNN on boundary-aligned (z, AR10 residual) + optional feature MLP branch.
    CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/deeplearning/bin/python probe/arm_seq.py tag USE_FEATS(0/1) [seeds] [epochs]"""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "probe"); import feats
tag = sys.argv[1]; UF = int(sys.argv[2]); NS = int(sys.argv[3]) if len(sys.argv) > 3 else 1; EP = int(sys.argv[4]) if len(sys.argv) > 4 else 25
PRE = 960; dev = "cuda"; t0 = time.time(); y = feats.y
Sa = torch.tensor(np.load("outputs/arms/seq_train.npy").astype(np.float32)); Sb = torch.tensor(np.load("outputs/arms/seq_test.npy").astype(np.float32))
if UF:
    X, Xt, _ = feats.load(); A = np.vstack([X.to_numpy(), Xt.to_numpy()]); miss = np.isnan(A); A = np.where(miss, np.nanmedian(A, 0), A)
    A = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9, random_state=0).fit_transform(A).astype(np.float32)
    Fa, Fb = torch.tensor(A[:len(y)]), torch.tensor(A[len(y):])
else:
    Fa, Fb = torch.zeros(len(y), 1), torch.zeros(len(Sb), 1)
class Blk(nn.Module):
    def __init__(s, c, d): super().__init__(); s.c = nn.Conv1d(c, c, 7, padding=3 * d, dilation=d); s.n = nn.BatchNorm1d(c); s.p = nn.Conv1d(c, c, 1)
    def forward(s, x): return x + s.p(F.gelu(s.n(s.c(x))))
class Net(nn.Module):
    def __init__(s, nf, c=64):
        super().__init__(); s.inp = nn.Conv1d(5, c, 1); s.b = nn.Sequential(*[Blk(c, d) for d in (1, 2, 4, 8, 16, 1, 2, 4)])
        s.f = nn.Sequential(nn.Dropout(.1), nn.Linear(nf, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(.4)) if UF else None
        s.h = nn.Sequential(nn.Linear(c * 8 + (256 if UF else 0), 128), nn.SiLU(), nn.Dropout(.3), nn.Linear(128, 1))
    def forward(s, x, f):
        m = x[:, 2:3]; e = x[:, 1:2]
        h = s.b(s.inp(torch.cat([x[:, :2], e.abs(), m, torch.cat([torch.ones_like(m[..., :PRE]), -torch.ones_like(m[..., PRE:])], -1) * m], 1))) * m
        mp, mq = m[..., :PRE], m[..., PRE:]
        def mean(h, w): return (h * w).sum(-1) / (w.sum(-1) + 1e-6)
        def sd(h, w): mu = mean(h, w); return (((h - mu[..., None]) ** 2 * w).sum(-1) / (w.sum(-1) + 1e-6) + 1e-6).sqrt()
        hp, hq = h[..., :PRE], h[..., PRE:]
        z = [mean(hp, mp), mean(hq, mq), sd(hp, mp), sd(hq, mq), mean(hq, mq) - mean(hp, mp),
             (hq[..., :40]).mean(-1), (hp[..., -40:]).mean(-1), (hq * mq - 1e4 * (1 - mq)).amax(-1) - (hp * mp - 1e4 * (1 - mp)).amax(-1)]
        z = torch.cat(z + ([s.f(f)] if UF else []), 1); return s.h(z).ravel()
def pred(m, S, Fe):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(S), 512): o.append(torch.sigmoid(m(S[i:i+512].to(dev), Fe[i:i+512].to(dev))).cpu())
    return torch.cat(o).numpy()
oof = np.zeros(len(y)); te = np.zeros(len(Sb))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    for s in range(NS):
        torch.manual_seed(s); a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=s)
        m = Net(Fa.shape[1]).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EP * ((len(a) + 127) // 128))
        yt = torch.tensor(y, dtype=torch.float32); best, bst = 0, None
        for ep in range(EP):
            m.train(); perm = np.random.default_rng(ep + 100 * s).permutation(a)
            for i in range(0, len(perm), 128):
                j = perm[i:i+128]; x = Sa[j].to(dev)
                opt.zero_grad(); F.binary_cross_entropy_with_logits(m(x, Fa[j].to(dev)), yt[j].to(dev)).backward(); opt.step(); sch.step()
            sc = roc_auc_score(y[b], pred(m, Sa[b], Fa[b]))
            if sc > best: best, bst = sc, {kk: v.clone() for kk, v in m.state_dict().items()}
        m.load_state_dict(bst); oof[va] += pred(m, Sa[va], Fa[va]) / NS; te += pred(m, Sb, Fb) / NS / 5
    print(f"fold {k}: {roc_auc_score(y[va], oof[va]):.4f} (inner best {best:.4f})  {time.time()-t0:.0f}s", flush=True)
print(f"{tag} OOF {roc_auc_score(y, oof):.4f}"); np.save(f"outputs/arms/{tag}_oof.npy", oof); np.save(f"outputs/arms/{tag}_test.npy", te)
