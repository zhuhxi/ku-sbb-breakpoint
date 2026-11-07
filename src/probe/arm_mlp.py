"""MLP arm on the production features (rank-gauss inputs), 5-fold same split as train.py, seeds averaged.
    CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/deeplearning/bin/python probe/arm_mlp.py [tag] [hidden] [drop] [seeds]"""
import sys, time, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "probe"); import feats
tag = sys.argv[1] if len(sys.argv) > 1 else "mlp"; H = int(sys.argv[2]) if len(sys.argv) > 2 else 512
DR = float(sys.argv[3]) if len(sys.argv) > 3 else .3; NS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
t0 = time.time(); X, Xt, y = feats.load()
A = np.vstack([X.to_numpy(), Xt.to_numpy()]); miss = np.isnan(A)
med = np.nanmedian(A, 0); A = np.where(miss, med, A)
A = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9, random_state=0).fit_transform(A)
mc = miss.mean(0) > .01; A = np.hstack([A, miss[:, mc].astype(np.float32)]).astype(np.float32)
Xa, Xb = A[:len(X)], A[len(X):]; dev = "cuda"
print(f"{Xa.shape[1]} inputs, prep {time.time()-t0:.0f}s", flush=True)
def net(d):
    return nn.Sequential(nn.Dropout(.1), nn.Linear(d, H), nn.BatchNorm1d(H), nn.SiLU(), nn.Dropout(DR),
                         nn.Linear(H, H // 2), nn.BatchNorm1d(H // 2), nn.SiLU(), nn.Dropout(DR), nn.Linear(H // 2, 1)).to(dev)
def pred(m, Z):
    m.eval()
    with torch.no_grad(): return torch.sigmoid(m(torch.tensor(Z, device=dev))).cpu().numpy().ravel()
oof = np.zeros(len(y)); te = np.zeros(len(Xb))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xa, y)):
    for s in range(NS):
        torch.manual_seed(s); a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=s)
        m = net(Xa.shape[1]); opt = torch.optim.AdamW(m.parameters(), float(__import__("os").environ.get("LR", 1e-3)), weight_decay=1e-2)
        xa, ya = torch.tensor(Xa[a], device=dev), torch.tensor(y[a], dtype=torch.float32, device=dev)
        best, bst, bad = 0, None, 0
        for ep in range(60):
            m.train(); perm = torch.randperm(len(a), device=dev)
            for i in range(0, len(a), 256):
                j = perm[i:i+256]; opt.zero_grad()
                nn.functional.binary_cross_entropy_with_logits(m(xa[j]).ravel(), ya[j]).backward(); opt.step()
            sc = roc_auc_score(y[b], pred(m, Xa[b]))
            if sc > best: best, bad, bst = sc, 0, {kk: v.clone() for kk, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 8: break
        m.load_state_dict(bst); oof[va] += pred(m, Xa[va]) / NS; te += pred(m, Xb) / NS / 5
    print(f"fold {k}: {roc_auc_score(y[va], oof[va]):.4f}  {time.time()-t0:.0f}s", flush=True)
print(f"OOF {roc_auc_score(y, oof):.4f}"); np.save(f"outputs/arms/{tag}_oof.npy", oof); np.save(f"outputs/arms/{tag}_test.npy", te)
