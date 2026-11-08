"""Domain forecaster trained on synthetic NO-BREAK fakes (tensor layout of seq_prep / sim_tensor): input = last W pre points
(z channel + mask), output = quantiles (.1,.5,.9) of the next H points. Then forecast-error features on real series (post first H
from pre; back-test on the last H pre points), same statistics as probe/chronos_feats.py. Prefix fc_.
    python probe/fc_net.py SIMDIR NFILE OUT [OUT_REV]"""
import sys, glob, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
SIMDIR, NF, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]; OUTR = sys.argv[4] if len(sys.argv) > 4 else None
PRE, W, H = 960, 256, 64; dev = "cuda"; torch.manual_seed(0); t0 = time.time(); QL = torch.tensor([.1, .5, .9], device=dev)
Xs = []
for f in sorted(glob.glob(SIMDIR + "/sim_*.npz"))[:NF]:
    z = np.load(f); X = z["X"][z["y"] == 0]; Xs.append(X[:, [0, 2], PRE - W - H:PRE + H])   # pre window + next H (from pre) and post H
X = torch.tensor(np.concatenate(Xs)); print("no-break fakes", len(X), f"{time.time()-t0:.0f}s", flush=True)
class Net(nn.Module):
    def __init__(s, c=64):
        super().__init__(); s.c = nn.Sequential(nn.Conv1d(2, c, 7, padding=3), nn.GELU(), nn.Conv1d(c, c, 7, padding=6, dilation=2), nn.GELU(), nn.Conv1d(c, c, 7, padding=12, dilation=4), nn.GELU(), nn.Conv1d(c, c, 7, padding=24, dilation=8), nn.GELU())
        s.h = nn.Sequential(nn.Flatten(), nn.Linear(c * W // 8, 512), nn.GELU(), nn.Linear(512, H * 3))
    def forward(s, x):   # x: (B, 2, W) -> (B, H, 3) quantiles in pre-normalised units
        return s.h(F.avg_pool1d(s.c(x), 8)).view(-1, H, 3)
def batch(Xb, off):   # off = 0: forecast post from pre ; off = H: back-test (window shifted left by H)
    inp = Xb[:, :, H - off:H - off + W].float(); tgt = Xb[:, 0, H - off + W:H - off + W + H].float(); msk = Xb[:, 1, H - off + W:H - off + W + H].float()
    return inp, tgt, msk
m = Net().to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4); EP = int(__import__("os").environ.get("EP", 2)); bs = 512
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EP * (len(X) // bs + 1))
for ep in range(EP):
    perm = torch.randperm(len(X)); tot = 0
    for i in range(0, len(X), bs):
        inp, tgt, msk = batch(X[perm[i:i + bs]].to(dev), H)   # train on the back-test position (all inside the pre/no-break regime)
        with torch.autocast("cuda", dtype=torch.bfloat16): q = m(inp).float()
        e = tgt[..., None] - q; loss = ((torch.maximum(QL * e, (QL - 1) * e)).sum(-1) * msk).sum() / (msk.sum() + 1)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item()
    print(f"epoch {ep}: pinball {tot / (len(X) // bs + 1):.4f} {time.time()-t0:.0f}s", flush=True)
def featurise(T):
    T = torch.tensor(T[:, [0, 2], PRE - W - H:PRE + H]); R = {}
    m.eval(); outs = {0: [], H: []}
    with torch.no_grad():
        for off in (0, H):
            for i in range(0, len(T), 2048):
                inp, tgt, msk = batch(T[i:i + 2048].to(dev), off); q = m(inp).float()
                outs[off].append(torch.cat([q, tgt[..., None], msk[..., None]], -1).cpu().numpy())
    D = {}
    for nm, off in (("post", 0), ("pre", H)):
        A = np.concatenate(outs[off]); lo, md, hi, act, mk = [A[..., j] for j in range(5)]; act = np.where(mk > 0, act, np.nan); w = hi - lo + 1e-3
        for k in (8, 16, 32, 64):
            e = act[:, :k] - md[:, :k]; D[f"fc_{nm}_e{k}"] = np.nanmean(np.abs(e) / w[:, :k], 1); D[f"fc_{nm}_b{k}"] = np.nanmean(e, 1) / (np.nanmean(w[:, :k], 1) + 1e-9)
            D[f"fc_{nm}_out{k}"] = np.nanmean((act[:, :k] < lo[:, :k]) | (act[:, :k] > hi[:, :k]), 1); D[f"fc_{nm}_sd{k}"] = np.log((np.nanstd(act[:, :k], 1) + 1e-6) / (np.nanmean(w[:, :k], 1) + 1e-6))
    for k in [c[8:] for c in list(D) if c.startswith("fc_post_")]: D[f"fc_diff_{k}"] = D[f"fc_post_{k}"] - D[f"fc_pre_{k}"]
    return D
ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
for sfx, out in (("", OUT), ("_rev", OUTR)):
    if out is None: continue
    T = np.concatenate([np.load(f"outputs/arms/seq_train{sfx}.npy"), np.load(f"outputs/arms/seq_test{sfx}.npy")])
    pd.DataFrame(featurise(T), index=pd.Index(ids, name="id")).to_parquet(out); print("wrote", out, f"{time.time()-t0:.0f}s")
