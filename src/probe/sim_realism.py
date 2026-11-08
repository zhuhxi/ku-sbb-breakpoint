"""Realism weights for synthetic pretraining data: embed all fakes and all real series (train+test, no labels) with a pretrained
CNN, fit a real-vs-fake LightGBM on PCA-32 embeddings (2-fold cross-fitted over the fakes), w = p/(1-p) clipped & normalised.
    CUDA_VISIBLE_DEVICES=0 python probe/sim_realism.py CKPT OUT.npy"""
import sys, numpy as np, torch, lightgbm as lgb
CK, OUT = sys.argv[1], sys.argv[2]; sys.argv = [sys.argv[0], "0"]; import seq_pretrain as sp
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
m = sp.Net().to(sp.dev); m.load_state_dict(torch.load(CK), strict=False); m.eval()
def embed(X):
    E = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 2048): E.append(m.emb(X[i:i+2048].to(sp.dev)).float().cpu())
    return torch.cat(E).numpy()
Er = embed(torch.cat([sp.Xr, torch.tensor(np.load("outputs/arms/seq_test.npy"))])); Ef = embed(sp.Xs)
mu, sd = Er.mean(0), Er.std(0) + 1e-6; pca = PCA(32, random_state=0).fit((np.vstack([Er, Ef[::10]]) - mu) / sd)
Pr, Pf = pca.transform((Er - mu) / sd), pca.transform((Ef - mu) / sd); W = np.zeros(len(Pf))
for a, b in KFold(2, shuffle=True, random_state=0).split(Pf):
    sub = a[np.random.default_rng(0).choice(len(a), min(len(a), 120000), replace=False)]
    d = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=31, verbose=-1, n_jobs=16).fit(np.r_[Pr, Pf[sub]], np.r_[np.ones(len(Pr)), np.zeros(len(sub))])
    p = d.predict_proba(Pf[b])[:, 1]; W[b] = p / (1 - p + 1e-6)
W = np.clip(W / np.median(W), 0, 10); W /= W.mean(); np.save(OUT, W.astype(np.float32))
print("weights q10/50/90/99", np.quantile(W, [.1, .5, .9, .99]).round(2), "ESS frac", round(W.sum() ** 2 / (W ** 2).sum() / len(W), 3))
