"""10 s check: 2nd-place window stats with FIXED windows (20/60/120 points) vs windows PROPORTIONAL to each series'
shorter segment (10/20/33% of min(n_pre, n_post)); same stats, transforms raw / z-score / |x|; 8000 series, fold 0, LGB lr .1."""
import sys, time
import lightgbm as lgb, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, "p2"); from feats import STATS, transforms, wstats
from scipy.stats import ttest_ind, ks_2samp, f as Fd, combine_pvalues
t0 = time.time()
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"].iloc[:8000]
S = pd.read_parquet("data/train_series.parquet"); S["id"] = S["id"].astype(str)
S = S[S.id.isin(set(L.index))].sort_values(["id", "time"])
def sym(v, n0):
    T = [transforms(v)[j] for j in (0, 1, 4)]; out = {}; m = min(n0, len(v) - n0); P = {"t": [], "f": [], "ks": []}
    for w, i in enumerate((20, 50, 100, 200, max(10, m // 2))):
        i = min(i, m)
        for j, t in enumerate(T):
            s = np.r_[np.nan, t[:-1]]; a, b = t[n0 - i:n0], t[n0:n0 + i]
            A, B = wstats(a, s[n0 - i:n0]), wstats(b, s[n0:n0 + i])
            for k in STATS: out[f"{k}_{j}_{w}_pre"], out[f"{k}_{j}_{w}_post"], out[f"{k}_{j}_{w}_d"] = A[k], B[k], B[k] - A[k]
            if j == 0:
                P["t"].append(ttest_ind(a, b, equal_var=False).pvalue); r = a.var(ddof=1) / b.var(ddof=1)
                P["f"].append(2 * min(Fd.cdf(r, i - 1, i - 1), 1 - Fd.cdf(r, i - 1, i - 1))); P["ks"].append(ks_2samp(a, b).pvalue)
    for k, p in P.items():
        p = np.clip(np.array(p), 1e-300, 1); out[f"{k}_fisher"] = -np.log10(combine_pvalues(p).pvalue + 1e-300); out[f"{k}_minp"] = -np.log10(p.min())
    return out
def one(v, n0, mode):
    if mode == "symmetric": return sym(v, n0)
    if mode == "fixed_pre_i_only":
        o = one(v, n0, "fixed"); return {k: x for k, x in o.items() if not (k.endswith("_2") or k.endswith("_3") or k.endswith("_2d") or k.endswith("_3d"))}
    T = [transforms(v)[j] for j in (0, 1, 4)]; out = {}
    m = min(n0, len(v) - n0)
    for w, i in enumerate((20, 60, 120) if mode == "fixed" else [max(5, int(f * m)) for f in (.1, .2, .33)]):
        for j, t in enumerate(T):
            s = np.r_[np.nan, t[:-1]]
            W = {p: wstats(t[sl], s[sl]) for p, sl in {0: slice(max(0, n0 - i), n0), 2: slice(max(0, n0 - 2 * i), n0),
                                                       3: slice(max(0, n0 - 3 * i), n0), 1: slice(n0, n0 + i)}.items()}
            for k in STATS:
                for p in (0, 1, 2, 3): out[f"{k}_{j}_{w}_{p}"] = W[p][k]
                for p in (0, 2, 3): out[f"{k}_{j}_{w}_{p}d"] = W[p][k] - W[1][k]
    return out
G = [(g["value"].to_numpy(float), int((g["period"] == 0).sum())) for _, g in S.groupby("id", sort=False)]
y = L.loc[S.id.unique()].to_numpy()
for mode in ("fixed", "fixed_pre_i_only"):
    X = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(one)(v, n0, mode) for v, n0 in G)).astype(float)
    tr, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    m = lgb.LGBMClassifier(learning_rate=.1, n_estimators=300, num_leaves=31, colsample_bytree=.5, verbose=-1, n_jobs=30).fit(X.iloc[tr], y[tr])
    print(f"{mode:12s} {X.shape[1]} cols  fold-0 AUC {roc_auc_score(y[va], m.predict_proba(X.iloc[va])[:, 1]):.4f}  {time.time() - t0:.1f}s", flush=True)
