"""Real vs fake (retuned three-bin): are hard-to-see positives as common in the fakes as in the real data?
Same 12 statistics (probe/gen_eval2.py) on 3000+3000 real and 3000+3000 fake series. Per domain: (a) label AUC of each statistic,
(b) a 5-fold LightGBM label model on the 12 stats inside the domain -> AUC and share of positives scored below the negatives'
median / 75th percentile (= 'hard positives', look like no-break)."""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, "probe"); import gen_eval2 as G
t0 = time.time(); NF = 3000
R = G.real().replace([np.inf, -np.inf], np.nan)
P = G.parse("outputs/factory/ku_gen_size_three-bin.json")
F = pd.DataFrame(Parallel(n_jobs=32, batch_size=32)(delayed(lambda s, y: dict(G.stats(*G.gen(P, 950_000 + 2*s + y, bool(y))[:2]), y=y))(s, y)
                                                     for s in range(NF) for y in (0, 1))).replace([np.inf, -np.inf], np.nan)
cols = [c for c in R if c != "y"]
def summ(D):
    p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, verbose=-1, n_jobs=32), D[cols], D.y, cv=5, method="predict_proba")[:, 1]
    neg, pos = p[D.y == 0], p[D.y == 1]
    return roc_auc_score(D.y, p), (pos < np.median(neg)).mean(), (pos < np.quantile(neg, .75)).mean()
for nm, D in (("real", R), ("fake", F)):
    a, h50, h75 = summ(D); print(f"{nm}: label AUC {a:.4f} | positives below neg median {h50:.3f} | below neg q75 {h75:.3f}")
print("per-statistic label AUC (real vs fake):")
for c in cols:
    ar = roc_auc_score(R.y, R[c].fillna(R[c].median())); af = roc_auc_score(F.y, F[c].fillna(F[c].median()))
    print(f"  {c:7s} real {ar:.3f}  fake {af:.3f}  fake-real {af-ar:+.3f}")
print(f"{time.time()-t0:.0f}s")
