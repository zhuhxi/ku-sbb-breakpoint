"""Generator debugging: real-vs-fake discriminator in fine-tuned Chronos-2 embedding space, fakeness broken down by EVERY hidden
generator setting (family, kind, noise, garch, trend, nuisances, lengths, sizes ...). Fakes made here with their info dicts.
    HF_HUB_OFFLINE=1 python probe/gen_disc.py PF.json N [CKPT]       (env UPD='dict(...)' overrides keys)"""
import sys, os, json, numpy as np, pandas as pd, torch
from joblib import Parallel, delayed
sys.path.insert(0, "probe"); from ku_gen import DEFAULT, gen
PF, N = sys.argv[1], int(sys.argv[2]); CK = sys.argv[3] if len(sys.argv) > 3 else "outputs/arms/c2mv_t.pt"
b = {k: (tuple(v) if isinstance(DEFAULT.get(k), tuple) else v) for k, v in json.load(open(PF)).items()}; P = dict(DEFAULT, **b)
if os.environ.get("UPD"): P.update(eval(os.environ["UPD"]))
PRE, POST, q = 960, 736, 10
def one(s):
    v, n0, inf = gen(P, 90000000 + s, bool(s % 2)); v = np.asarray(v, float); out = np.zeros((3, PRE + POST), np.float32)
    inf = dict(inf, n=len(v), r0=n0 / len(v), y=s % 2)
    if n0 > PRE or len(v) - n0 > POST or n0 < 2 * q: return None
    x0 = v[:n0]; med = np.median(x0); mad = 1.4826 * np.median(np.abs(x0 - med)) + 1e-8; z = (v - med) / mad
    Zl = np.column_stack([np.ones(len(z) - q)] + [z[q - j - 1:len(z) - j - 1] for j in range(q)])
    c = np.linalg.lstsq(Zl[:n0 - q], z[q:n0], rcond=None)[0]; e = np.r_[np.zeros(q), z[q:] - Zl @ c]; e /= e[q:n0].std() + 1e-8
    lo = PRE - n0; out[0, lo:PRE + len(v) - n0] = np.clip(z, -30, 30); out[1, lo:PRE + len(v) - n0] = np.clip(e, -30, 30); out[2, lo:PRE + len(v) - n0] = 1; out[1, lo:lo + q] = 0
    return out, inf
res = [r for r in Parallel(n_jobs=24, batch_size=64)(delayed(one)(s) for s in range(N)) if r is not None]
Xf = np.stack([r[0] for r in res]); I = pd.DataFrame([r[1] for r in res])
os.environ.setdefault("WIN", "512"); sys.argv = [sys.argv[0], "x", "0", "0", "x"]
exec(open("probe/c2mv_ft.py").read().split("net = C2Cls().to(dev)")[0]); net = C2Cls().to(dev); net.load_state_dict(torch.load(CK), strict=False); net.eval()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
def emb(X):
    o = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 256): o.append(net.emb(torch.tensor(X[i:i + 256], device=dev)).float().cpu().numpy())
    return np.concatenate(o)
Er = emb(win(np.load("outputs/arms/seq_train.npy").astype(np.float32))); Ef = emb(win(Xf)); yr = pd.read_csv("data/train_labels.csv").label.values
E = np.vstack([Er, Ef]); d = np.r_[np.zeros(len(Er)), np.ones(len(Ef))]
Pc = PCA(64, random_state=0).fit_transform((E - E.mean(0)) / (E.std(0) + 1e-6))
p = cross_val_predict(LogisticRegression(C=.1, max_iter=3000), Pc, d, cv=5, method="predict_proba")[:, 1]
pr, pf = p[:len(Er)], p[len(Er):]; yf = I.y.to_numpy()
print(f"real-vs-fake AUC {roc_auc_score(d, p):.4f} | no-break {roc_auc_score(np.r_[np.zeros((yr==0).sum()), np.ones((yf==0).sum())], np.r_[pr[yr==0], pf[yf==0]]):.4f} | break {roc_auc_score(np.r_[np.zeros((yr==1).sum()), np.ones((yf==1).sum())], np.r_[pr[yr==1], pf[yf==1]]):.4f}")
base = pf.mean(); rows = []
for c in I.columns:
    v = I[c]
    if v.dtype == object or v.nunique() <= 6: g = v.fillna("none").astype(str)
    else: g = pd.qcut(v.rank(method="first"), 4, labels=False).astype(str) + ":" + pd.qcut(v, 4, duplicates="drop").astype(str).str[:18]
    for k, s in pd.Series(pf).groupby(g.values):
        if len(s) >= 200: rows.append((c, k, len(s), s.mean() - base))
R = pd.DataFrame(rows, columns=["key", "value", "n", "fake"]).sort_values("fake", ascending=False)
pd.set_option("display.width", 200); print(R.head(14).to_string(index=False)); print("... most REAL-looking:"); print(R.tail(8).to_string(index=False))
I["fake"] = pf; I.to_parquet("outputs/arms/gen_disc_last.parquet"); np.save("outputs/arms/gen_disc_real.npy", pr)
