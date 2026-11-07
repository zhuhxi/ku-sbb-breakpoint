"""Boundary-straddling features (event study: real disturbance starts ~6 steps BEFORE n0). Reference = pre part EXCLUDING the
last G points. AR(10) fit on the clean pre; standardized error z in windows [-G,0), [0,10), [10,30), [30,80), [80,end);
level shift, log sd ratio, acf1 change of the post part vs the clean pre; and the transition window alone. -> outputs/factory/gap.parquet"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "probe"); import feats
import os; G = int(os.environ.get("G", 8)); q = 10
def one(v, n0):
    x0 = v[:n0 - G]; mu, sd = x0.mean(), x0.std() + 1e-9; z = (v - mu) / sd
    Zl = np.column_stack([np.ones(len(z) - q)] + [z[q - j - 1:len(z) - j - 1] for j in range(q)])
    m = n0 - G - q; c = np.linalg.lstsq(Zl[:m], z[q:n0 - G], rcond=None)[0]; e = z[q:] - Zl @ c; s = e[:m].std() + 1e-9; e = e / s
    E = lambda a, b: np.mean(e[max(a + n0 - q, 0):min(b + n0 - q, len(e))] ** 2) if b + n0 - q > a + n0 - q and a + n0 - q < len(e) else np.nan
    post = z[n0:]; tr = z[n0 - G:n0]
    ac = lambda u: np.corrcoef(u[:-1], u[1:])[0, 1] if len(u) > 3 else np.nan
    d = dict(gp_eT=E(-G, 0), gp_e0=E(0, 10), gp_e1=E(10, 30), gp_e2=E(30, 80), gp_e3=E(80, 10**6), gp_eA=E(0, 10**6),
             gp_lvl=post.mean(), gp_lsd=np.log(post.std() + 1e-9), gp_acf=ac(post) - ac(x0), gp_trl=tr.mean(), gp_trsd=np.log(tr.std() + 1e-9),
             gp_lvl20=post[:20].mean(), gp_lsd20=np.log(post[:20].std() + 1e-9))
    d["gp_e0_e3"] = np.log((d["gp_e0"] + .05) / (d["gp_e3"] + .05)); d["gp_eT_e0"] = np.log((d["gp_eT"] + .05) / (d["gp_e0"] + .05))
    for k in ("gp_eT", "gp_e0", "gp_e1", "gp_e2", "gp_e3", "gp_eA"): d[k] = np.log(d[k] + .05)
    return d
rows = {}
for path in ("data/train_series.parquet", "data/test_series.parquet"):
    S = pd.read_parquet(path); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"], kind="stable")
    for sid, g in S.groupby("id", sort=False):
        v = g["value"].to_numpy(float); rows[sid] = one(v, int((g["period"].to_numpy() == 0).sum()))
D = pd.DataFrame.from_dict(rows, orient="index"); D.index.name = "id"; D.to_parquet(f"outputs/factory/gap{G}.parquet")
from sklearn.metrics import roc_auc_score
T = D.reindex(feats.lab.index)
print({c: round(max(a := roc_auc_score(feats.y, T[c].fillna(T[c].median())), 1 - a), 3) for c in T})
