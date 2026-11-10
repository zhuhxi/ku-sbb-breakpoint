"""Real-vs-fake discriminator in fine-tuned Chronos-2 (two-variate) embedding space: which generator settings look least real?
    python probe/disc_c2.py CKPT SIMFILE N"""
import sys, os, numpy as np, pandas as pd, torch
CK, SF, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.environ.setdefault("WIN", "512"); sys.argv = [sys.argv[0], "x", "0", "0", "x"]
exec(open("probe/c2mv_ft.py").read().split("net = C2Cls().to(dev)")[0]); net = C2Cls().to(dev); net.load_state_dict(torch.load(CK), strict=False); net.eval()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
def emb(X):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 256): out.append(net.emb(torch.tensor(X[i:i + 256], device=dev)).float().cpu().numpy())
    return np.concatenate(out)
z = np.load(SF); ok = z["y"] >= 0; Xf = z["X"][ok][:N]; Mf = z["M"][ok][:N]; yf = z["y"][ok][:N]
Er = emb(win(np.load("outputs/arms/seq_train.npy"))); Ef = emb(win(Xf)); yr = pd.read_csv("data/train_labels.csv").label.values
E = np.vstack([Er, Ef]); d = np.r_[np.zeros(len(Er)), np.ones(len(Ef))]
P = PCA(64, random_state=0).fit_transform((E - E.mean(0)) / (E.std(0) + 1e-6))
p = cross_val_predict(LogisticRegression(C=.1, max_iter=2000), P, d, cv=5, method="predict_proba")[:, 1]
print("real-vs-fake AUC all", round(roc_auc_score(d, p), 4), "| no-break", round(roc_auc_score(np.r_[np.zeros((yr == 0).sum()), np.ones((yf == 0).sum())], np.r_[p[:len(Er)][yr == 0], p[len(Er):][yf == 0]]), 4),
      "| break", round(roc_auc_score(np.r_[np.zeros((yr == 1).sum()), np.ones((yf == 1).sum())], np.r_[p[:len(Er)][yr == 1], p[len(Er):][yf == 1]]), 4))
pf = p[len(Er):]; base = pf.mean(); print("mean fakeness of fakes", round(base, 4))
names = {0: "kind", 1: "family", 5: "ntype"}
for j, nm in names.items():
    s = pd.Series(pf).groupby(Mf[:, j].astype(int)).agg(["mean", "size"]); print(nm, {int(k): (round(v["mean"] - base, 3), int(v["size"])) for k, v in s.iterrows()})
for j in (2, 3, 4):
    q = pd.qcut(Mf[:, j], 4, duplicates="drop"); s = pd.Series(pf).groupby(q, observed=True).mean() - base; print("size col", j, s.round(3).to_dict())
for j in (6, 7, 8): print("nuis col", j, "on", round(pf[Mf[:, j] > 0].mean() - base, 3), "n", int((Mf[:, j] > 0).sum()))
np.save("outputs/arms/disc_fakeness.npy", np.c_[pf, Mf, yf])
# real side: which real series look most real/fake
np.save("outputs/arms/disc_real_p.npy", p[:len(Er)])
