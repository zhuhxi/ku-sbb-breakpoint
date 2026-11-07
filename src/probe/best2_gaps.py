"""Re-measure the two suspected generator gaps on the CURRENT best2 generator (probe/ku_gen.py + ku_gen_best2.json).
1) isolated spikes (same definition as probe/spike_scan.py): per 1000 points before / after the boundary, real vs fake, per label.
2) random walks: real RW series (genA struct 5) vs fake RW (family RW), per label; KS distance of simple increment statistics
   (sd ratio after/before, lag-1 acf of increments, kurtosis of increments, mean increment shift in sd units) + a C2ST
   (LightGBM real-vs-fake on those statistics, 5-fold AUC; .5 = indistinguishable). WN as the reference family.
    python probe/best2_gaps.py"""
import json, sys, time
import lightgbm as lgb, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kurtosis, ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, "probe")
from ku_gen import DEFAULT, gen
t0 = time.time()
B = {k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v) for k, v in json.load(open("outputs/factory/ku_gen_best2.json")).items()}
P = dict(DEFAULT, **B)


def spikes(x, n0, T=6):
    p = 5
    A = np.column_stack([np.ones(len(x) - p)] + [x[p - k:len(x) - k] for k in range(1, p + 1)])
    e = np.r_[np.zeros(p), x[p:] - A @ np.linalg.lstsq(A, x[p:], rcond=None)[0]]
    pe = e[p:n0]; z = (e - np.median(pe)) / (1.4826 * np.median(np.abs(pe - np.median(pe))) + 1e-12); z[:p] = 0
    a = np.abs(z); iso = (a > T) & (np.r_[0, a[:-1]] < 3) & (np.r_[a[1:], 0] < 3)
    return iso[:n0].sum() / n0 * 1000, iso[n0:].sum() / (len(x) - n0) * 1000


def incr(x, n0):
    d = np.diff(x); b, a = d[:n0 - 1], d[n0:]
    ac = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]
    return dict(sd_ratio=np.log(a.std() / b.std()), acf_b=ac(b), acf_a=ac(a), kurt_b=kurtosis(b), kurt_a=kurtosis(a), dmean=(a.mean() - b.mean()) / b.std())


def one(x, n0):
    s = spikes(x, n0); return dict(sp_pre=s[0], sp_post=s[1], **incr(x, n0))


L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
A = pd.read_parquet("outputs/factory/genA.parquet"); A.index = A.index.astype(str); st = A.g_struct.reindex(L.index)
S = pd.read_parquet("data/train_series.parquet"); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"])
items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
R = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(one)(x, n0) for _, x, n0 in items), index=[k for k, _, _ in items])
R["label"] = L.loc[R.index].to_numpy(); R["fam"] = st.loc[R.index].map({0: "WN", 5: "RW"}).fillna("other").to_numpy()


def fk(s):
    x, n0, h = gen(P, 9_000_000 + s, bool(s % 2)); return dict(one(x, n0), label=s % 2, fam=h["family"])


F = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(fk)(s) for s in range(24000)))
print(f"measured {len(R)} real, {len(F)} fakes  {time.time() - t0:.0f}s\n1) isolated spikes |z|>6 per 1000 points")
for y in (0, 1):
    r, f = R[R.label == y], F[F.label == y]
    print(f"  label {y}: pre  real {r.sp_pre.mean():.3f} fake {f.sp_pre.mean():.3f} ({f.sp_pre.mean() / r.sp_pre.mean() - 1:+.0%}) | "
          f"post real {r.sp_post.mean():.3f} fake {f.sp_post.mean():.3f} ({f.sp_post.mean() / r.sp_post.mean() - 1:+.0%})")
print("2) increment statistics, KS distance real vs fake (and C2ST AUC), per family and label")
st_cols = ["sd_ratio", "acf_b", "acf_a", "kurt_b", "kurt_a", "dmean"]
for fam in ("RW", "WN"):
    for y in (0, 1):
        r = R[(R.fam == fam) & (R.label == y)][st_cols].replace([np.inf, -np.inf], np.nan).dropna()
        f = F[(F.fam == fam) & (F.label == y)][st_cols].replace([np.inf, -np.inf], np.nan).dropna()
        ks = {c: ks_2samp(r[c], f[c]).statistic for c in st_cols}
        n = min(len(r), len(f)); Xc = pd.concat([r.sample(n, random_state=0), f.sample(n, random_state=0)]); yc = np.r_[np.zeros(n), np.ones(n)]
        auc = roc_auc_score(yc, cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, verbose=-1, n_jobs=30), Xc, yc, cv=5, method="predict_proba")[:, 1])
        print(f"  {fam} label {y} (real {len(r)}, fake {len(f)}): C2ST AUC {auc:.3f} | KS " + " ".join(f"{c} {v:.3f}" for c, v in ks.items()))
print(f"{time.time() - t0:.0f}s")
