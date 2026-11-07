"""Where does the generated NO-BREAK series differ from real label-0 series? Real label 0 vs gen(ok4, brk=False)
(probe/calib_break.py), ~40 whole-series statistics on the per-series standardized values (+ shell / length stats),
per-statistic real-vs-fake AUC and a LightGBM C2ST with importances.

    python probe/nb_diff.py [N] [MENU]
"""
import importlib.util
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
MENU = sys.argv[2] if len(sys.argv) > 2 else "ok4"
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


def acf(z, k):
    return float(np.dot(z[k:], z[:-k]) / len(z))


def feats(x, n0):
    n = len(x)
    z = (x - x.mean()) / (x.std() + 1e-12)
    d = np.diff(z)
    d = (d - d.mean()) / (d.std() + 1e-12)
    q = np.quantile(z, [.01, .05, .25, .5, .75, .95, .99])
    P = np.abs(np.fft.rfft(z)) ** 2
    P = P[1:] / P[1:].sum()
    t = np.arange(n) / n
    r2 = np.corrcoef(t, z)[0, 1] ** 2
    s = np.sign(z - np.median(z))
    runs = np.diff(np.flatnonzero(np.r_[1, np.diff(s) != 0, 1]))
    a2 = z * z - 1
    r = dict(n=n, frac0=n0 / n, mean=x.mean(), lsd=np.log(x.std()), skew=skew(z), kurt=kurtosis(z),
             **{f"q{p}": v for p, v in zip((1, 5, 25, 50, 75, 95, 99), q)},
             **{f"acf{k}": acf(z, k) for k in (1, 2, 3, 5, 10, 20)},
             dacf1=acf(d, 1), dacf2=acf(d, 2), dkurt=kurtosis(d), sqacf1=acf(a2, 1) / (a2.var() + 1e-12),
             absacf1=acf(np.abs(z) - np.abs(z).mean(), 1) / (np.abs(z).var() + 1e-12),
             pk=P.max(), pk_f=P.argmax() / len(P), plow=P[:max(len(P) // 20, 1)].sum(), sflat=np.exp(np.mean(np.log(P + 1e-15))) / P.mean(),
             trend_r2=r2, maxabs=np.abs(z).max(), runmax=runs.max() / n, runmean=runs.mean(),
             first=z[0], last=z[-1], half_lsd=np.log(z[n // 2:].std() / (z[:n // 2].std() + 1e-12)),
             half_dmean=z[n // 2:].mean() - z[:n // 2].mean(), ties=1 - len(np.unique(x)) / n)
    return r


t0 = time.time()
L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
ids = L.loc[L.label == 0, "id"].to_numpy()[:N]
S = pd.read_parquet("data/train_series.parquet")
S = S[S["id"].astype(str).isin(set(ids))].sort_values(["id", "time"])
real = [(g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for _, g in S.groupby("id", sort=False)]
B = cb.BREAKS[MENU]
fake = Parallel(n_jobs=30, batch_size=32)(delayed(cb.gen)(B, 10_000 + s, False, True) for s in range(N))
F = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(feats)(x, n0) for x, n0 in real + list(fake))).astype(float)
y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
auc = pd.Series({c: roc_auc_score(y, F[c].fillna(0)) for c in F})
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, verbose=-1, n_jobs=30)
for nm, cols in (("no shell (mean, lsd)", [c for c in F if c not in ("mean", "lsd")]), ("shell only", ["mean", "lsd"]),
                 ("quantiles+moments", [c for c in F if c[0] == "q" or c in ("skew", "kurt", "maxabs")]),
                 ("acf/spectrum", [c for c in F if "acf" in c or c in ("pk", "pk_f", "plow", "sflat", "runmax", "runmean")]),
                 ("length", ["n", "frac0"])):
    pp = cross_val_predict(lgb.LGBMClassifier(**P), F[cols], y, cv=5, method="predict_proba")[:, 1]
    print(f"  C2ST {nm:22s} {roc_auc_score(y, pp):.3f}")
p = cross_val_predict(lgb.LGBMClassifier(**P), F, y, cv=5, method="predict_proba")[:, 1]
imp = pd.Series(lgb.LGBMClassifier(**P).fit(F, y).booster_.feature_importance("gain"), F.columns)
print(f"menu {MENU}: {len(real)} real vs {len(fake)} fake, C2ST AUC {roc_auc_score(y, p):.3f}  ({time.time() - t0:.1f}s)")
top = (auc - .5).abs().sort_values(ascending=False).index[:12]
R = pd.DataFrame({"AUC(fake>real)": auc[top].round(3), "real median": F.loc[y == 0, top].median().round(3),
                  "fake median": F.loc[y == 1, top].median().round(3), "gain share": (imp[top] / imp.sum()).round(3)})
print(R.to_string())
