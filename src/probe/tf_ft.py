"""Fine-tune the Chronos-2 encoder (+ small head) as a break classifier on synthetic fakes (tensor layout of sim_tensor / seq_prep:
channel 0 = pre-median/MAD z, channel 2 = mask; window PRE-W .. PRE+W, NaN outside the mask). Reports zero-shot AUC on real train.
    python probe/c2_ft.py SIMDIR FOFF NFILE OUT [INIT]"""
import sys, glob, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

SIMDIR, FOFF, NF, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; INIT = sys.argv[5] if len(sys.argv) > 5 else None
PRE, W, dev = 960, int(__import__("os").environ.get("WIN", 256)), "cuda"; t0 = time.time(); torch.manual_seed(0)
def win(T):   # (n, 3, L) float16 tensor array -> (n, 2W) float32 with NaN outside the mask
    z = T[:, 0, PRE - W:PRE + W].astype(np.float32); mk = T[:, 2, PRE - W:PRE + W] > 0; z[~mk] = np.nan; return z
class C2Cls(nn.Module):   # TimesFM-2.5 backbone (causal decoder, 32-pt patches); same interface as the Chronos-2 classifier
    def __init__(s):
        super().__init__(); import timesfm
        s.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(timesfm.TimesFM_2p5_200M_torch.DEFAULT_REPO_ID).model.to(dev)
        D = s.m.tokenizer(torch.zeros(1, 1, 64, device=dev)).shape[-1]; s.D = D
        s.h = nn.Sequential(nn.Linear(D * 4, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 1))
        s.aux = nn.Sequential(nn.Linear(D * 4, 256), nn.GELU(), nn.Linear(256, 20))
    def emb(s, x):
        B = len(x); msk = torch.isnan(x); xx = torch.where(msk, torch.zeros_like(x), x).clamp(-30, 30)
        P = xx.view(B, -1, 32); Mk = msk.view(B, -1, 32)
        (_, out, _, _), _ = s.m(P, Mk); h = W // 32
        pre, post = out[:, :h].mean(1), out[:, h:].mean(1); return torch.cat([pre, post, post - pre, out[:, -1]], 1)
    def forward(s, x): return s.h(s.emb(x)).squeeze(-1)
net = C2Cls().to(dev)
if INIT: net.load_state_dict(torch.load(INIT), strict=False)
Xs, ys, Ms = [], [], []; MT = int(__import__("os").environ.get("MT", 0))
for f in sorted(glob.glob(SIMDIR + "/sim_*.npz"))[FOFF:FOFF + NF]:
    z = np.load(f); ok = z["y"] >= 0; Xs.append(win(z["X"][ok])); ys.append(z["y"][ok]); Ms.append(z["M"][ok])
X = np.concatenate(Xs); Y = np.concatenate(ys).astype(np.float32); MM = np.concatenate(Ms).astype(np.float32); print("fakes", len(X), f"{time.time()-t0:.0f}s", flush=True)
Xr = win(np.load("outputs/arms/seq_train.npy")); yr = pd.read_csv("data/train_labels.csv").label.values
def zs():
    net.eval(); o = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(Xr), 1024): o.append(net(torch.tensor(Xr[i:i + 1024], device=dev)).float().cpu().numpy())
    net.train(); return roc_auc_score(yr, np.concatenate(o))
BLR = float(__import__("os").environ.get("BLR", 2e-5)); opt = torch.optim.AdamW([{"params": net.m.parameters(), "lr": BLR}, {"params": net.h.parameters(), "lr": 1e-3}], weight_decay=1e-2); bs = 128
steps = len(X) // bs; sch = torch.optim.lr_scheduler.OneCycleLR(opt, [BLR, 1e-3], total_steps=steps, pct_start=.1)
perm = np.random.default_rng(FOFF).permutation(len(X)); net.train()
for k in range(steps):
    j = perm[k * bs:(k + 1) * bs]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        e = net.emb(torch.tensor(X[j], device=dev)); loss = F.binary_cross_entropy_with_logits(net.h(e).squeeze(-1).float(), torch.tensor(Y[j], device=dev))
        if MT:
            M = torch.tensor(MM[j], device=dev); A = net.aux(e).float()
            loss = loss + .5 * (F.cross_entropy(A[:, :5], M[:, 0].long()) + F.cross_entropy(A[:, 5:11], M[:, 1].long()) + F.mse_loss(A[:, 11:14], torch.log1p(5 * M[:, 2:5]))
                                + F.cross_entropy(A[:, 14:17], M[:, 5].long().clamp(0, 2)) + F.binary_cross_entropy_with_logits(A[:, 17:20], M[:, 6:9]))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    if k % 400 == 0: print(f"step {k}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
torch.save(net.state_dict(), OUT); print(f"zero-shot AUC on real train {zs():.4f}  {time.time()-t0:.0f}s")
