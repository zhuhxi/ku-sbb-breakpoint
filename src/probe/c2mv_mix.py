"""Fold-clean fine-tune of the two-variate Chronos-2 break classifier on REAL labels MIXED with genR10 fakes (fakes act as a regulariser).
StratifiedKFold 5 seed 42; inner 10% early stop (checked NCHK times per epoch). Saves outputs/arms/{TAG}_f{k}.npz (va, test; fwd+rev logits).
    python probe/c2mv_mix.py INIT TAG folds...   env NFAKE (fakes per real copy ratio), BLR, EP, NCHK"""
import sys, os, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
CK0, TAG, FOLDS = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3:]]; os.environ.setdefault("WIN", "512"); sys.argv = [sys.argv[0], "x", "0", "0", "x"]
exec(open("probe/c2mv_ft.py").read().split("net = C2Cls().to(dev)")[0])
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
y = pd.read_csv("data/train_labels.csv").label.values.astype(np.float32)
Xf = win(np.load("outputs/arms/seq_train.npy")); Xb = win(np.load("outputs/arms/seq_train_rev.npy"))
Tf = win(np.load("outputs/arms/seq_test.npy")); Tb = win(np.load("outputs/arms/seq_test_rev.npy"))
NFAKE = float(os.environ.get("NFAKE", .5)); BLR = float(os.environ.get("BLR", 5e-6)); EP = int(os.environ.get("EP", 1)); NCHK = int(os.environ.get("NCHK", 3))
z = np.load(os.environ.get("FAKE", "/mnt/disk2/zhuhx/kagglecompetition/simR10/sim_6.npz")); ok = z["y"] >= 0; FX = win(z["X"][ok]); FY = z["y"][ok].astype(np.float32); del z
print("loaded", f"{time.time()-t0:.0f}s", flush=True)
def pred(net, A):
    net.eval(); o = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(A), 512): o.append(net(torch.tensor(A[i:i + 512], device=dev)).float().cpu().numpy())
    net.train(); return np.concatenate(o)
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    if k not in FOLDS: continue
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k); net = C2Cls().to(dev); net.load_state_dict(torch.load(CK0), strict=False)
    opt = torch.optim.AdamW([{"params": net.m.parameters(), "lr": BLR}, {"params": net.h.parameters(), "lr": 1e-4}], weight_decay=1e-2)
    nf = int(NFAKE * 2 * len(a)); fi = np.random.default_rng(k).choice(len(FX), min(nf, len(FX)), replace=False)
    A2 = np.concatenate([Xf[a], Xb[a], FX[fi]]); Y2 = np.concatenate([y[a], y[a], FY[fi]])
    sc0 = roc_auc_score(y[b], pred(net, Xf[b]) + pred(net, Xb[b])); best, bst = sc0, {kk: v.clone() for kk, v in net.state_dict().items()}; hist = []
    for ep in range(EP):
        perm = np.random.default_rng(ep + 10 * k).permutation(len(A2)); chk = set(int(len(perm) / 64 * (c + 1) / NCHK) for c in range(NCHK))
        for s, i in enumerate(range(0, len(perm), 64)):
            j = perm[i:i + 64]
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = F.binary_cross_entropy_with_logits(net(torch.tensor(A2[j], device=dev)).float(), torch.tensor(Y2[j], device=dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if s + 1 in chk:
                sc = roc_auc_score(y[b], pred(net, Xf[b]) + pred(net, Xb[b])); hist.append(round(sc, 4))
                if sc > best: best, bst = sc, {kk: v.clone() for kk, v in net.state_dict().items()}
    v0 = None
    net.load_state_dict(bst); pv = pred(net, Xf[va]) + pred(net, Xb[va]); pt = pred(net, Tf) + pred(net, Tb)
    np.savez(f"outputs/arms/{TAG}_f{k}.npz", va=pv, test=pt)
    print(f"fold {k}: inner {sc0:.4f} -> {hist} best {best:.4f} | val {roc_auc_score(y[va], pv):.4f} {time.time()-t0:.0f}s", flush=True)
