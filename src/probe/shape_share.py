"""Break class only: is the 11% 'shape' break kind real? Compare p_kind shape share 0 / .05 / .11(current) / .20 (others rescaled),
same objective as probe/size_ln.py (C2ST + AD/100 + mean q90/q99 error in real-IQR units), 3 fake seed sets each."""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from scipy.stats import anderson_ksamp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, "probe"); import gen_eval2 as G
t0 = time.time(); NF = 1500
R = G.real(); r = R[R.y == 1].drop(columns="y").replace([np.inf, -np.inf], np.nan)
iqr = (r.quantile(.75) - r.quantile(.25)).replace(0, 1)
def ev(P, seed):
    F = pd.DataFrame(Parallel(n_jobs=32, batch_size=32)(delayed(lambda s: G.stats(*G.gen(P, seed + 2*s + 1, True)[:2]))(s) for s in range(NF))).replace([np.inf, -np.inf], np.nan)
    ad = np.mean([anderson_ksamp([r[c].dropna().to_numpy(), F[c].dropna().to_numpy()]).statistic for c in r])
    q = np.mean([abs(r[c].quantile(a) - F[c].quantile(a)) / iqr[c] for c in r for a in (.9, .99)])
    X = pd.concat([r, F]); t = np.r_[np.zeros(len(r)), np.ones(len(F))]
    auc = roc_auc_score(t, cross_val_predict(lgb.LGBMClassifier(n_estimators=150, learning_rate=.05, num_leaves=15, verbose=-1, n_jobs=32), X, t, cv=5, method="predict_proba")[:, 1])
    return dict(c2st=auc, ad=ad, q=q)
tb = G.parse("outputs/factory/ku_gen_size_three-bin.json"); pk = np.asarray(tb["p_kind"], float)
for sh in (0.0, .05, pk[3], .20):
    o = pk[:3] / pk[:3].sum() * (1 - sh); P = dict(tb, p_kind=[*o, sh])
    rs = [ev(P, sd) for sd in (810_000, 820_000, 830_000)]
    print(f"shape share {sh:.3f}: " + "  ".join(f"{k} {np.mean([x[k] for x in rs]):.4f}±{np.std([x[k] for x in rs]):.4f}" for k in rs[0]) + f"  {time.time()-t0:.0f}s", flush=True)
