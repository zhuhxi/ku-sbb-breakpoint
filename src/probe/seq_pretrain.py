"""Pretrain the dilated 1D CNN of probe/arm_seq.py (sequence-only) on synthetic aligned tensors; report zero-shot AUC on all
real train series (no real labels used for training). Saves outputs/arms/seqpre.pt
    CUDA_VISIBLE_DEVICES=0 python probe/seq_pretrain.py EPOCHS [C]"""
import sys, time, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
EP = int(sys.argv[1]); C = int(sys.argv[2]) if len(sys.argv) > 2 else 64; PRE = 960; dev = "cuda"; t0 = time.time(); torch.manual_seed(0)
Z = [np.load(f) for f in sorted(glob.glob(__import__("os").environ.get("SIMDIR", "/mnt/disk2/zhuhx/kagglecompetition/simt") + "/sim_*.npz"))[int(__import__("os").environ.get("FOFF", 0)):int(__import__("os").environ.get("FOFF", 0)) + int(__import__("os").environ.get("NFILE", 99))]] if sys.argv[1] != "0" else []
Xs = torch.tensor(np.concatenate([z["X"] for z in Z])) if Z else None; ys = torch.tensor(np.concatenate([z["y"] for z in Z]).astype(np.float32)) if Z else None
MT = int(__import__("os").environ.get("MT", 0)); Ms = torch.tensor(np.concatenate([z["M"] for z in Z])) if MT and Z else None
Xr = torch.tensor(np.load("outputs/arms/seq_train.npy")); yr = np.loadtxt("data/train_labels.csv", delimiter=",", skiprows=1, usecols=1)
class Blk(nn.Module):
    def __init__(s, c, d): super().__init__(); s.c = nn.Conv1d(c, c, 7, padding=3 * d, dilation=d); s.n = nn.BatchNorm1d(c); s.p = nn.Conv1d(c, c, 1)
    def forward(s, x): return x + s.p(F.gelu(s.n(s.c(x))))
class Net(nn.Module):
    def __init__(s, c=C):
        super().__init__(); s.inp = nn.Conv1d(5, c, 1); s.b = nn.Sequential(*[Blk(c, d) for d in (1, 2, 4, 8, 16, 32, 1, 2, 4, 8)])
        s.h = nn.Sequential(nn.Linear(c * 8, 128), nn.SiLU(), nn.Dropout(.2), nn.Linear(128, 1))
        s.aux = nn.Sequential(nn.Linear(c * 8, 128), nn.SiLU(), nn.Linear(128, 5 + 6 + 3 + 3 + 3))   # kind, family, |log_k| |lvl_d| |dep_d|, ntype, garch/decoy/trend
    def emb(s, x):
        x = x.float(); m = x[:, 2:3]; e = x[:, 1:2]
        h = s.b(s.inp(torch.cat([x[:, :2], e.abs(), m, torch.cat([torch.ones_like(m[..., :PRE]), -torch.ones_like(m[..., PRE:])], -1) * m], 1))) * m
        mp, mq = m[..., :PRE], m[..., PRE:]; hp, hq = h[..., :PRE], h[..., PRE:]
        mean = lambda h, w: (h * w).sum(-1) / (w.sum(-1) + 1e-6)
        sd = lambda h, w: ((((h - mean(h, w)[..., None]) ** 2) * w).sum(-1) / (w.sum(-1) + 1e-6) + 1e-6).sqrt()
        return torch.cat([mean(hp, mp), mean(hq, mq), sd(hp, mp), sd(hq, mq), mean(hq, mq) - mean(hp, mp), hq[..., :40].mean(-1), hp[..., -40:].mean(-1),
                          (hq * mq - 1e4 * (1 - mq)).amax(-1) - (hp * mp - 1e4 * (1 - mp)).amax(-1)], 1)
    def forward(s, x): return s.h(s.emb(x)).ravel()
class TNet(Net):
    """Patch transformer encoder (ARCH=tf): 5 input channels -> patches of 16 -> 4-layer encoder (d 128) -> c channels per token;
    same masked pre/post pooling and heads as the CNN, so seq_emb.py works unchanged."""
    def __init__(s, c=C, d=128, P=16):
        super().__init__(c); del s.b; s.P = P; s.pe = nn.Conv1d(5, d, P, stride=P); s.pos = nn.Parameter(torch.zeros(1, 256, d)); s.seg = nn.Embedding(2, d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, .1, batch_first=True, norm_first=True), 4); s.out = nn.Linear(d, c)
    def emb(s, x):
        x = x.float(); m = x[:, 2:3]; e = x[:, 1:2]; P = s.P; Q = PRE // P; W = max(1, 40 // P)
        z = torch.cat([x[:, :2], e.abs(), m, torch.cat([torch.ones_like(m[..., :PRE]), -torch.ones_like(m[..., PRE:])], -1) * m], 1)
        t = s.pe(z).transpose(1, 2); mt = F.avg_pool1d(m, P, P)[:, 0] > .5; T = t.shape[1]
        sg = s.seg((torch.arange(T, device=t.device) >= Q).long())[None]
        h = s.out(s.enc(t + s.pos[:, :T] + sg, src_key_padding_mask=~mt)).transpose(1, 2) * mt[:, None].float(); m = mt[:, None].float()
        mp, mq = m[..., :Q], m[..., Q:]; hp, hq = h[..., :Q], h[..., Q:]
        mean = lambda h, w: (h * w).sum(-1) / (w.sum(-1) + 1e-6)
        sd = lambda h, w: ((((h - mean(h, w)[..., None]) ** 2) * w).sum(-1) / (w.sum(-1) + 1e-6) + 1e-6).sqrt()
        return torch.cat([mean(hp, mp), mean(hq, mq), sd(hp, mp), sd(hq, mq), mean(hq, mq) - mean(hp, mp), hq[..., :W].mean(-1), hp[..., -W:].mean(-1),
                          (hq * mq - 1e4 * (1 - mq)).amax(-1) - (hp * mp - 1e4 * (1 - mp)).amax(-1)], 1)
if __import__("os").environ.get("ARCH") == "tf": Net = TNet
def predict(m, X):
    m.eval(); o = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 1024): o.append(m(X[i:i+1024].to(dev)).float().cpu())
    return torch.cat(o).numpy()
if __name__ == "__main__":
    import os
    m = Net().to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2); bs = 256
    if os.environ.get("INIT"): m.load_state_dict(torch.load(os.environ["INIT"]))   # continue a previous pretraining run
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, float(os.environ.get("LRMAX", 2e-3)), total_steps=EP * (len(Xs) // bs + 1))
    print(f"fakes {len(Xs)}, load {time.time()-t0:.0f}s", flush=True)
    for ep in range(EP):
        m.train(); perm = torch.multinomial(torch.tensor(np.load(os.environ["WTS"])), len(Xs), replacement=True) if os.environ.get("WTS") else torch.randperm(len(Xs))
        for i in range(0, len(Xs), bs):
            j = perm[i:i+bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = m.emb(Xs[j].to(dev)); loss = F.binary_cross_entropy_with_logits(m.h(e).ravel(), ys[j].to(dev))
                if MT:
                    M = Ms[j].to(dev); A = m.aux(e).float()
                    loss = loss + .5 * (F.cross_entropy(A[:, :5], M[:, 0].long()) + F.cross_entropy(A[:, 5:11], M[:, 1].long()) + F.mse_loss(A[:, 11:14], torch.log1p(5 * M[:, 2:5]))
                                        + F.cross_entropy(A[:, 14:17], M[:, 5].long().clamp(0, 2)) + F.binary_cross_entropy_with_logits(A[:, 17:20], M[:, 6:9]))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        pz = predict(m, Xr); print(f"epoch {ep}: zero-shot AUC on real train {roc_auc_score(yr, pz):.4f}  {time.time()-t0:.0f}s", flush=True)
        torch.save(m.state_dict(), os.environ.get("OUT", f"outputs/arms/seqpre{'_mt' if MT else ''}_c{C}.pt"))
