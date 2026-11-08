"""Fold-clean fine-tune of the synthetic-pretrained Chronos-2 classifier on REAL labels (StratifiedKFold 5 seed 42; inner 10% early stop,
EP epochs). Saves outputs/arms/{TAG0}_f{k}.npz (va, test); prediction = mean of fwd and reversed windows.
    python probe/c2_real.py INIT TAG folds..."""
import sys, os, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
CKPT0, TAG0, FOLDS = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3:]]; os.environ.setdefault("WIN", "512"); sys.argv = [sys.argv[0], "x", "0", "0", "x"]
exec(open("probe/c2_ft.py").read().split("net = C2Cls().to(dev)")[0])
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
y = pd.read_csv("data/train_labels.csv").label.values.astype(np.float32)
Xf = win(np.load("outputs/arms/seq_train.npy")); Xb = win(np.load("outputs/arms/seq_train_rev.npy"))
Tf = win(np.load("outputs/arms/seq_test.npy")); Tb = win(np.load("outputs/arms/seq_test_rev.npy")); EP = int(os.environ.get("EP", 3))
def pred(net, A):
    net.eval(); o = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(A), 512): o.append(net(torch.tensor(A[i:i + 512], device=dev)).float().cpu().numpy())
    net.train(); return np.concatenate(o)
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    if k not in FOLDS: continue
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k); net = C2Cls().to(dev); net.load_state_dict(torch.load(CKPT0))
    opt = torch.optim.AdamW([{"params": net.m.parameters(), "lr": 1e-5}, {"params": net.h.parameters(), "lr": 3e-4}], weight_decay=1e-2)
    A2 = np.concatenate([Xf[a], Xb[a]]); Y2 = np.concatenate([y[a], y[a]])   # reversed copies as augmentation
    best = roc_auc_score(y[b], pred(net, Xf[b]) + pred(net, Xb[b])); bst = {kk: v.clone() for kk, v in net.state_dict().items()}; b0 = best
    for ep in range(EP):
        perm = np.random.default_rng(ep + 10 * k).permutation(len(A2))
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = F.binary_cross_entropy_with_logits(net(torch.tensor(A2[j], device=dev)).float(), torch.tensor(Y2[j], device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sc = roc_auc_score(y[b], pred(net, Xf[b]) + pred(net, Xb[b]))
        if sc > best: best, bst = sc, {kk: v.clone() for kk, v in net.state_dict().items()}
    net.load_state_dict(bst); pv = pred(net, Xf[va]) + pred(net, Xb[va]); pt = pred(net, Tf) + pred(net, Tb)
    np.savez(f"outputs/arms/{TAG0}_f{k}.npz", va=pv, test=pt); print(f"fold {k}: inner {b0:.4f}->{best:.4f} val {roc_auc_score(y[va], pv):.4f} {time.time()-t0:.0f}s", flush=True)
