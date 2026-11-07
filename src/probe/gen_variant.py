"""Compare generator variants one at a time. Each arg: SPEC = json path [+key=val,...] (gen_eval2.parse; tuples as a:b).
Per variant: break-class realism vs real (C2ST, AD, q90/q99 error; mean of 3 seed sets) and near-boundary signals:
label AUC of near20 / lvar (real .588 / .609) and median start ratio of label-1 series (real .980). Env CLS=0 scores the realism on the no-break class instead.
    python probe/gen_variant.py BASE.json BASE.json+spread_target=innov"""
import os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from scipy.stats import anderson_ksamp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, "probe"); import gen_eval2 as G
t0 = time.time(); NF = 1500
R = G.real(); CLS = int(os.environ.get("CLS", 1)); r = R[R.y == CLS].drop(columns="y").replace([np.inf, -np.inf], np.nan)
iqr = (r.quantile(.75) - r.quantile(.25)).replace(0, 1)
def one(P, seed, y):
    x, n0 = G.gen(P, seed, bool(y))[:2]; d = np.diff(x)[n0:]
    return dict(G.stats(x, n0), start=d[:20].var() / d[20:].var(), y=y)
def ev(P, seed):
    F = pd.DataFrame(Parallel(n_jobs=32, batch_size=32)(delayed(one)(P, seed + 2*s + y, y) for s in range(NF) for y in (0, 1))).replace([np.inf, -np.inf], np.nan)
    f1 = F[F.y == CLS][r.columns]
    ad = np.mean([anderson_ksamp([r[c].dropna().to_numpy(), f1[c].dropna().to_numpy()]).statistic for c in r])
    q = np.mean([abs(r[c].quantile(a) - f1[c].quantile(a)) / iqr[c] for c in r for a in (.9, .99)])
    X = pd.concat([r, f1]); t = np.r_[np.zeros(len(r)), np.ones(len(f1))]
    auc = roc_auc_score(t, cross_val_predict(lgb.LGBMClassifier(n_estimators=150, learning_rate=.05, num_leaves=15, verbose=-1, n_jobs=32), X, t, cv=5, method="predict_proba")[:, 1])
    return dict(c2st=auc, ad=ad, q=q, near20=roc_auc_score(F.y, F.near20.fillna(0)), lvar=roc_auc_score(F.y, F.lvar.fillna(0)), start1=F[F.y == 1].start.median())
for spec in sys.argv[1:]:
    base, _, extra = spec.partition("+")
    P = G.parse(base + ("+" + ",".join(kv for kv in extra.split(",") if ":" not in kv) if extra else ""))
    for kv in filter(lambda s: ":" in s, extra.split(",")):
        k, v = kv.split("="); P[k] = tuple(float(u) for u in v.split(":"))
    rs = [ev(P, sd) for sd in (810_000, 820_000, 830_000)]
    print(f"{(extra or 'base'):40s} " + "  ".join(f"{k} {np.mean([x[k] for x in rs]):.4f}" for k in rs[0]) + f"  {time.time()-t0:.0f}s", flush=True)
