"""Interpretable real-vs-fake test: LightGBM on ~40 hand statistics (pre / post / change), per label class; prints AUC and the
statistics that separate real from fake most (gain), with real vs fake quantiles.   python probe/gen_c2st.py PF.json [UPD]"""
import sys, json, os, numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "probe"); from ku_gen import DEFAULT, gen
b = {k: (tuple(v) if isinstance(DEFAULT.get(k), tuple) else v) for k, v in json.load(open(sys.argv[1])).items()}; P = dict(DEFAULT, **b)
if len(sys.argv) > 2: P.update(eval(sys.argv[2]))
def ac(v, k=1): v = v - v.mean(); return (v[k:] * v[:-k]).sum() / ((v * v).sum() + 1e-12)
def seg(v, p):
    d = np.diff(v); z = (v - v.mean()) / (v.std() + 1e-12); zd = (d - d.mean()) / (d.std() + 1e-12)
    return {f"{p}acf1": ac(v), f"{p}acf2": ac(v, 2), f"{p}acf10": ac(v, 10), f"{p}dacf1": ac(d), f"{p}ku": (z ** 4).mean() - 3, f"{p}sk": (z ** 3).mean(),
            f"{p}dku": (zd ** 4).mean() - 3, f"{p}sq1": ac(z ** 2), f"{p}sq10": ac(z ** 2, 10), f"{p}vr": np.log(d.var() / (v.var() + 1e-12) + 1e-12),
            f"{p}trend": np.corrcoef(np.arange(len(v)), v)[0, 1], f"{p}half": np.log(np.std(v[len(v)//2:]) / (np.std(v[:len(v)//2]) + 1e-12) + 1e-12),
            f"{p}absz3": np.mean(np.abs(z) > 3), f"{p}c0": np.mean(np.abs(z) < .3), **(ext(v, p) if EXT else {})}
EXT = bool(os.environ.get("EXT"))
def ext(v, p):   # periodogram peak (sine detector): share of variance at the strongest period in 10-40, that period, its 2nd harmonic share
    v = v - v.mean(); n = len(v); F = np.abs(np.fft.rfft(v, 8 * n)) ** 2; f = np.fft.rfftfreq(8 * n)
    m = (f > 1 / 40) & (f < 1 / 10); i = np.argmax(np.where(m, F, 0)); tot = (v @ v) * 8 * n / 2 + 1e-12
    j = min(2 * i, len(F) - 1)
    return {f"{p}pk": F[i] / tot * 2 / 8, f"{p}pkper": 1 / f[i], f"{p}pk2": F[j] / (F[i] + 1e-12)}
def stats(x, n0):
    x = np.asarray(x, float); a, b_ = x[:n0], x[n0:]; sd = np.std(np.diff(a)) / np.sqrt(2) + 1e-12
    o = dict(seg(a, "a_"), **seg(b_, "b_")); o.update(n=len(x), r0=n0 / len(x), lsd=np.log(np.std(np.diff(b_)) / (np.std(np.diff(a)) + 1e-12)),
        lvl=(np.median(b_[:50]) - np.median(a[-50:])) / sd, near=(b_[:10].mean() - a[-10:].mean()) / sd, rng=np.log((x.max() - x.min()) / sd))
    return o
S = pd.read_parquet("data/train_series.parquet"); S["id"] = S.id.astype(str); S = S.sort_values(["id", "time"])
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id").label
CACHE = "outputs/arms/gen_c2st_real%s.parquet" % ("_ext" if EXT else "")
if os.path.exists(CACHE): R = pd.read_parquet(CACHE)
else: R = pd.DataFrame([dict(stats(g.value.to_numpy(), int((g.period == 0).sum())), y=L[k]) for k, g in S.groupby("id")]); R.to_parquet(CACHE)
SEED = int(os.environ.get("SEED", 99000000))
F = pd.DataFrame(Parallel(n_jobs=24, batch_size=64)(delayed(lambda s: dict(stats(*gen(P, SEED + s, bool(s % 2))[:2]), y=s % 2))(s) for s in range(24000)))
if os.environ.get("SUB"): R, F = R.query(os.environ["SUB"]), F.query(os.environ["SUB"])   # e.g. SUB='a_acf1>.9 and a_vr<-2.5' (random-walk-like)
cols = [c for c in R if c != "y"]
for y in (0, 1):
    X = pd.concat([R[R.y == y][cols], F[F.y == y][cols]]).replace([np.inf, -np.inf], np.nan); t = np.r_[np.zeros((R.y == y).sum()), np.ones((F.y == y).sum())]
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=31, verbose=-1, n_jobs=12)
    p = cross_val_predict(m, X, t, cv=5, method="predict_proba")[:, 1]; m.fit(X, t)
    imp = pd.Series(m.booster_.feature_importance("gain"), cols).sort_values(ascending=False)
    print(f"label {y}: real-vs-fake AUC {roc_auc_score(t, p):.4f}", flush=True)
    if os.environ.get("QUIET"): continue
    for c in imp.index[:6]:
        print(f"   {c:8s} gain {imp[c]/imp.sum():.2f}  real {np.round(np.nanquantile(X[c][t==0],[.05,.5,.95]),3)}  fake {np.round(np.nanquantile(X[c][t==1],[.05,.5,.95]),3)}")
