"""Fine-tune the synthetic-pretrained CNN on real data, fold-clean (StratifiedKFold 5, seed 42; inner 10% early stopping).
    CUDA_VISIBLE_DEVICES=0 python probe/seq_finetune.py CKPT TAG EPOCHS LR folds..."""
import sys, time, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.argv_ft = sys.argv[:]; sys.argv = [sys.argv[0], "0"]
import seq_pretrain as sp
a = sys.argv_ft; CK, TAG, EP, LR = a[1], a[2], int(a[3]), float(a[4]); FOLDS = [int(v) for v in a[5:]] if len(a) > 5 else range(5)
Xr, yr = sp.Xr, sp.yr.astype(np.float32); Xt = torch.tensor(np.load("outputs/arms/seq_test.npy")); t0 = time.time()
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yr)), yr)):
    if k not in FOLDS: continue
    a_, b_ = train_test_split(tr, test_size=.1, stratify=yr[tr], random_state=k)
    m = sp.Net().to(sp.dev); m.load_state_dict(torch.load(CK), strict=False); opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-2)
    best, bst = roc_auc_score(yr[b_], sp.predict(m, Xr[b_])), {kk: v.clone() for kk, v in m.state_dict().items()}; b0 = best
    for ep in range(EP):
        m.train(); perm = np.random.default_rng(ep).permutation(a_)
        for i in range(0, len(perm), 128):
            j = perm[i:i+128]
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = F.binary_cross_entropy_with_logits(m(Xr[j].to(sp.dev)), torch.tensor(yr[j], device=sp.dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sc = roc_auc_score(yr[b_], sp.predict(m, Xr[b_]))
        if sc > best: best, bst = sc, {kk: v.clone() for kk, v in m.state_dict().items()}
    m.load_state_dict(bst); pv = sp.predict(m, Xr[va]); pt = sp.predict(m, Xt)
    np.savez(f"outputs/arms/{TAG}_f{k}.npz", va=pv, test=pt); print(f"fold {k}: inner {b0:.4f}->{best:.4f}  val {roc_auc_score(yr[va], pv):.4f}  {time.time()-t0:.0f}s", flush=True)
