"""Balanced generator check (replaces mean KS alone): on a held-out real sample (3000 per class, disjoint from the search cache),
12 statistics = probe/gen_search.py st (8) + near20, whole-post lvar, post/pre log sd of the AR(10) pre-model errors, cv of
|AR(10) residuals| (the C2ST gaps). Per class: mean KS, mean Anderson-Darling (tail-weighted), mean |q90| and |q99| error in
real-IQR units, and a C2ST AUC (LightGBM real vs fake on the 12 statistics, 5-fold; .5 = indistinguishable).
    python probe/gen_eval2.py NAME=JSON_OR_best2[+key=val,...] ...     e.g.  best2   best2+size_mode=hn,lvl_cs=.8
"""
import json, sys, time, warnings
import lightgbm as lgb, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import anderson_ksamp, ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
warnings.filterwarnings("ignore")
sys.path.insert(0, "probe")
import gen_search as gs
from ku_gen import DEFAULT, gen
NR, NF, p = 3000, 3000, 10


def arfit(s):
    X = np.column_stack([np.ones(len(s) - p)] + [s[p - k:len(s) - k] for k in range(1, p + 1)])
    b = np.linalg.lstsq(X, s[p:], rcond=None)[0]; return b, s[p:] - X @ b


def stats(x, n0):
    out = gs.st(x, n0)
    b, e = arfit(x[:n0]); sd = e.std(); full = x[n0 - p:]
    X = np.column_stack([np.ones(len(full) - p)] + [full[p - k:len(full) - k] for k in range(1, p + 1)])
    o = (full[p:] - X @ b) / sd; _, ew = arfit(x); a = np.abs(ew)
    out.update(near20=np.mean(o[:20] ** 2), lvar=np.log(np.mean(o ** 2)), lsdr=np.log(o.std()), cv=a.std() / a.mean())
    return out


def real():
    L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    first = set(L[L == 0].sample(2500, random_state=0).index) | set(L[L == 1].sample(2500, random_state=0).index)
    rest = L[~L.index.isin(first)]
    keep = set(rest[rest == 0].sample(NR, random_state=11).index) | set(rest[rest == 1].sample(NR, random_state=11).index)
    S = pd.read_parquet("data/train_series.parquet"); S["id"] = S["id"].astype(str); S = S[S.id.isin(keep)].sort_values(["id", "time"])
    return pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(lambda k, v, n0: dict(stats(v, n0), y=L[k]))(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum()))
                                                           for k, g in S.groupby("id", sort=False)))


def parse(spec):
    base, _, extra = spec.partition("+")
    src = "outputs/factory/ku_gen_best2.json" if base == "best2" else base
    P = dict(DEFAULT, **{k: (tuple(v) if isinstance(DEFAULT.get(k), tuple) else v) for k, v in json.load(open(src)).items()})
    for kv in filter(None, extra.split(",")):
        k, v = kv.split("=")
        try: v = float(v)
        except ValueError: pass
        P[k] = v
    return P


def evaluate(P, R, seed=700_000):
    res = {}
    for y in (0, 1):
        F = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(lambda s: stats(*gen(P, seed + 2 * s + y, bool(y))[:2]))(s) for s in range(NF)))
        r = R[R.y == y].drop(columns="y")
        F, r = F.replace([np.inf, -np.inf], np.nan), r.replace([np.inf, -np.inf], np.nan)
        ks = np.mean([ks_2samp(r[c].dropna(), F[c].dropna()).statistic for c in r])
        ad = np.mean([anderson_ksamp([r[c].dropna().to_numpy(), F[c].dropna().to_numpy()]).statistic for c in r])
        iqr = (r.quantile(.75) - r.quantile(.25)).replace(0, np.nan)
        q90 = np.nanmean(np.abs(F.quantile(.9) - r.quantile(.9)) / iqr); q99 = np.nanmean(np.abs(F.quantile(.99) - r.quantile(.99)) / iqr)
        X = pd.concat([r, F]); t = np.r_[np.zeros(len(r)), np.ones(len(F))]
        auc = roc_auc_score(t, cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, verbose=-1, n_jobs=30), X, t, cv=5, method="predict_proba")[:, 1])
        res[y] = dict(ks=ks, ad=ad, q90=q90, q99=q99, c2st=auc)
    return res


if __name__ == "__main__":
    t0 = time.time(); R = real(); print(f"real {len(R)} {time.time() - t0:.0f}s")
    for spec in sys.argv[1:]:
        name, _, sp = spec.partition("=") if "=" in spec.split("+")[0] else (spec, "", spec)
        res = evaluate(parse(sp), R)
        print(f"{name:28s} " + " | ".join(f"y{y}: KS {v['ks']:.4f} AD {v['ad']:6.2f} q90 {v['q90']:.3f} q99 {v['q99']:.3f} C2ST {v['c2st']:.3f}" for y, v in res.items()) + f"  {time.time() - t0:.0f}s", flush=True)
