"""Break class only: retuned three-bin vs + transient post-break shock (shock_p / shock_a / shock_tau). Objective = C2ST + AD/100 + mean q90/q99 error (real-IQR units)."""
import sys, time, json
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from scipy.stats import anderson_ksamp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, "probe"); import gen_eval2 as G
t0 = time.time(); NT = int(sys.argv[1]); NF = 1500
R = G.real(); r = R[R.y == 1].drop(columns="y").replace([np.inf, -np.inf], np.nan)
iqr = (r.quantile(.75) - r.quantile(.25)).replace(0, 1)
def ev(P, seed):
    F = pd.DataFrame(Parallel(n_jobs=32, batch_size=32)(delayed(lambda s: G.stats(*G.gen(P, seed + 2*s + 1, True)[:2]))(s) for s in range(NF))).replace([np.inf, -np.inf], np.nan)
    ad = np.mean([anderson_ksamp([r[c].dropna().to_numpy(), F[c].dropna().to_numpy()]).statistic for c in r])
    q = np.mean([abs(r[c].quantile(a) - F[c].quantile(a)) / iqr[c] for c in r for a in (.9, .99)])
    X = pd.concat([r, F]); t = np.r_[np.zeros(len(r)), np.ones(len(F))]
    auc = roc_auc_score(t, cross_val_predict(lgb.LGBMClassifier(n_estimators=150, learning_rate=.05, num_leaves=15, verbose=-1, n_jobs=32), X, t, cv=5, method="predict_proba")[:, 1])
    return dict(c2st=auc, ad=ad, q=q, obj=auc + ad/100 + q)
tb = G.parse("outputs/factory/ku_gen_size_three-bin.json"); rng = np.random.default_rng(1); U = lambda a, b: float(rng.uniform(a, b))
def draw():
    a = U(.2, 1.5); tau = U(2, 12); return dict(tb, shock_p=U(.1, .8), shock_a=(a, a + U(0, 1.5)), shock_tau=(tau, tau + U(0, 15)))
cands = [draw() for _ in range(NT)]
sc = [ev(P, 700_000) for P in cands]
for P, s in sorted(zip(cands, sc), key=lambda z: z[1]["obj"])[:3]:
    print({k: np.round(P[k], 3).tolist() for k in ("shock_p", "shock_a", "shock_tau")}, {k: round(v, 4) for k, v in s.items()}, flush=True)
best = cands[int(np.argmin([s["obj"] for s in sc]))]
print(f"search {time.time()-t0:.0f}s; re-score on 3 new seed sets:")
for nm, P in (("three-bin retuned", tb), ("+ shock best", best)):
    rs = [ev(P, sd) for sd in (810_000, 820_000, 830_000)]
    print(f"  {nm:18s}", {k: round(np.mean([x[k] for x in rs]), 4) for k in rs[0]}, flush=True)
json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()}, open("outputs/factory/ku_gen_shock.json", "w"), indent=1)
print(f"{time.time()-t0:.0f}s")
