"""Self-supervised break-injection detector on REAL series (no labels): positives = injected copies (ssl/inj_*.npz), negatives =
the untouched series (train+test). Init from a pretrained CNN. Then features for every real series: detector logit + PCA32 of
the pooled embedding -> outputs/factory/seqssl.parquet. Also prints the detector logit's AUC vs the real train labels (diagnostic only)."""
import sys, glob, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
EP = int(sys.argv[1]); INIT = sys.argv[2]; sys.argv = [sys.argv[0], "0"]; import seq_pretrain as sp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
t0 = time.time(); torch.manual_seed(0)
Xo = torch.cat([sp.Xr, torch.tensor(np.load("outputs/arms/seq_test.npy"))]); Xi = torch.cat([torch.tensor(np.load(f)["X"]) for f in sorted(glob.glob("/mnt/disk2/zhuhx/kagglecompetition/ssl/inj_*.npz"))])
X = torch.cat([Xo.repeat(len(Xi) // len(Xo), 1, 1), Xi]); Y = torch.cat([torch.zeros(len(Xi)), torch.ones(len(Xi))])
m = sp.Net().to(sp.dev); m.load_state_dict(torch.load(INIT), strict=False); opt = torch.optim.AdamW(m.parameters(), 5e-4, weight_decay=1e-2); bs = 256
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * (len(X) // bs + 1)); yr = sp.yr
for ep in range(EP):
    m.train(); perm = torch.randperm(len(X))
    for i in range(0, len(X), bs):
        j = perm[i:i+bs]
        with torch.autocast("cuda", dtype=torch.bfloat16): loss = F.binary_cross_entropy_with_logits(m(X[j].to(sp.dev)), Y[j].to(sp.dev))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    p = sp.predict(m, sp.Xr); print(f"epoch {ep}: detector logit vs real labels AUC {roc_auc_score(yr, p):.4f}  {time.time()-t0:.0f}s", flush=True)
m.eval(); E, Lg = [], []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(0, len(Xo), 1024): e = m.emb(Xo[i:i+1024].to(sp.dev)); E.append(e.float().cpu()); Lg.append(m.h(e).ravel().float().cpu())
E = torch.cat(E).numpy(); Lg = torch.cat(Lg).numpy(); P = PCA(32, random_state=0).fit_transform((E - E.mean(0)) / (E.std(0) + 1e-6))
ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
D = pd.DataFrame(P, index=pd.Index(ids, name="id"), columns=[f"se_ssl_pc{i}" for i in range(32)]); D["se_ssl_logit"] = Lg; D.to_parquet("outputs/factory/seqssl.parquet"); print(D.shape)
