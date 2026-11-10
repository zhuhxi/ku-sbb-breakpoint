"""Per-segment NOISE classifier (oracle test: knowing the pre noise type/df is worth +.007 AUC on fakes).
AR(5) fitted on the pre segment whitens the whole series; for each segment the standardized residuals give distribution stats.
LightGBM models trained on genR10 fakes (bags a,b; pre segments, + post segments of series without a shape break, whose post noise
= pre noise) predict noise type (gauss / t / bimodal), 1/df and GARCH; applied to the pre and post segments of real train+test.
    python probe/seg_noise.py OUT.parquet"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from scipy import stats
t0 = time.time(); OUT = sys.argv[1]; P = 5; import os; REVR = bool(os.environ.get("REVSER")); QV = bool(os.environ.get("QV")); P = int(os.environ.get("ARP", 5))


def sstats(z):
    z = z[np.isfinite(z)]
    if len(z) < 20: return [np.nan] * len(NAMES)
    m, s = np.median(z), np.std(z) + 1e-12; u = (z - np.mean(z)) / s; q = np.quantile(u, [.01, .05, .25, .75, .95, .99]); iqr = q[3] - q[2] + 1e-12
    ku = stats.kurtosis(u); sk = stats.skew(u); a2 = u[1:] ** 2; b2 = u[:-1] ** 2
    out = [ku, abs(sk), (q[5] - q[0]) / iqr, (q[4] - q[1]) / iqr, np.mean(np.abs(u)), np.mean(np.abs(u) > 2.5), np.mean(np.abs(u) > 3.5),
           (sk ** 2 + 1) / (ku + 3), np.mean(np.abs(u) < .25), np.corrcoef(a2, b2)[0, 1], np.max(np.abs(u)), np.log(len(u)), stats.kurtosis(np.abs(u))]
    if QV:   # finer shape: |u| quantile profile (median-scaled), tail index, |u| autocorrelation at lags 2-5
        au = np.abs(u); md = np.median(au) + 1e-12; aq = np.quantile(au, [.1, .25, .75, .9, .95, .98, .99]) / md
        k = max(5, len(au) // 20); top = np.sort(au)[-k - 1:]; hill = np.mean(np.log(top[1:] / top[0] + 1e-12))
        acs = [np.corrcoef(au[l:], au[:-l])[0, 1] for l in (2, 3, 5)]
        out += list(aq) + [hill] + acs + [np.mean(u ** 6) / (np.mean(u ** 2) ** 3 + 1e-12)]
    return out


NAMES = ["ku", "ask", "q99r", "q95r", "mabs", "f25", "f35", "bimc", "fnear", "arch", "mx", "logn", "kuabs"] + (["aq10", "aq25", "aq75", "aq90", "aq95", "aq98", "aq99", "hill", "aac2", "aac3", "aac5", "m6"] if QV else [])


def one(sid, v, p):
    n0 = int((p == 0).sum()); x = v - v[:n0].mean()
    X = np.column_stack([x[P - k - 1:len(x) - k - 1] for k in range(P)]); yv = x[P:]
    c, *_ = np.linalg.lstsq(np.c_[np.ones(n0 - P), X[:n0 - P]], yv[:n0 - P], rcond=None)
    r = yv - np.c_[np.ones(len(yv)), X] @ c
    return sid, sstats(r[:n0 - P]), sstats(r[n0 - P + 1:])   # skip the boundary residual


def run(df, real=False):
    g = [(sid, x.value.to_numpy(np.float64), x.period.to_numpy()) for sid, x in df.groupby("id", sort=False)]
    if REVR and real: g = [(s_, v[::-1].copy(), 1 - p[::-1]) for s_, v, p in g]   # time-reversed REAL series (fakes stay forward)
    res = Parallel(n_jobs=24, batch_size=64)(delayed(one)(*a) for a in g)
    ids = [a for a, _, _ in res]; return (pd.DataFrame([b for _, b, _ in res], index=ids, columns=NAMES), pd.DataFrame([c for _, _, c in res], index=ids, columns=NAMES))


FPre, FPost, Ms = [], [], []
for b in os.environ.get("BAGS", "ab"):
    a_, c_ = run(pd.read_parquet(f"outputs/aug/series_simr10{b}.parquet")); FPre.append(a_); FPost.append(c_); Ms.append(pd.read_parquet(f"outputs/aug/meta_simr10{b}.parquet"))
FPre, FPost, M = pd.concat(FPre), pd.concat(FPost), pd.concat(Ms); M = M.loc[FPre.index]; print("fake stats", FPre.shape, f"{time.time()-t0:.0f}s", flush=True)
keep = M["shape"].to_numpy() == 0   # post noise = pre noise
XT = pd.concat([FPre, FPost[keep]]).to_numpy(np.float32); MT = pd.concat([M, M[keep]])
tg = {"ty": MT.ntype.to_numpy().astype(int), "idf": MT.inv_df.to_numpy(), "ga": MT.garch.to_numpy().astype(int)}
n = len(XT); rng = np.random.default_rng(0); te = rng.random(n) < .15
mods = {}
for k, t in tg.items():
    obj = dict(objective="multiclass", num_class=3) if k == "ty" else dict(objective="regression") if k == "idf" else dict(objective="binary")
    m = lgb.LGBMRegressor(**obj, n_estimators=400, learning_rate=.05, num_leaves=63, min_child_samples=50, verbose=-1, n_jobs=24) if k == "idf" else lgb.LGBMClassifier(**obj, n_estimators=400, learning_rate=.05, num_leaves=63, min_child_samples=50, verbose=-1, n_jobs=24)
    m.fit(XT[~te], t[~te]); pr = m.predict(XT[te]) if k == "idf" else m.predict_proba(XT[te])
    sc = np.corrcoef(pr, t[te])[0, 1] if k == "idf" else (pr.argmax(1) == t[te]).mean()
    print(f"  {k}: held-out {'corr' if k == 'idf' else 'acc'} {sc:.3f}", flush=True); mods[k] = m.fit(XT, t)
R1, R2 = run(pd.concat([pd.read_parquet("data/train_series.parquet"), pd.read_parquet("data/test_series.parquet")]), real=True); print("real stats", R1.shape, f"{time.time()-t0:.0f}s", flush=True)
def pred(F):
    X = F.to_numpy(np.float32); ty = mods["ty"].predict_proba(X)
    return pd.DataFrame({"g": ty[:, 0], "t": ty[:, 1], "b": ty[:, 2], "idf": mods["idf"].predict(X), "ga": mods["ga"].predict_proba(X)[:, 1]}, index=F.index)
a, b = pred(R1), pred(R2)
D = pd.concat([a.add_prefix("sn_pre_"), b.add_prefix("sn_post_")], axis=1)
D["sn_tv"] = (a[["g", "t", "b"]] - b[["g", "t", "b"]]).abs().sum(1) / 2; D["sn_didf"] = b.idf - a.idf; D["sn_dga"] = b.ga - a.ga
D["sn_ce"] = -(a[["g", "t", "b"]].values * np.log(b[["g", "t", "b"]].values + 1e-6)).sum(1)
for c in ("ku", "q99r", "f35", "fnear", "arch"): D[f"sn_d{c}"] = R2[c] - R1[c]
D.index.name = "id"; D.to_parquet(OUT); print(D.shape, f"{time.time()-t0:.0f}s")
