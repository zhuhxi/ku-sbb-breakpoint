"""Small real-vs-fake test (5-10 s): 1500 real train series vs 1500 fakes, cheap per-series features,
with and without the outer shell. Random-walk-like series (pre lag-1 corr > .9) dropped on both sides.

    python probe/shell_c2st.py [MENU ...]
"""
import importlib.util
import time

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

t0 = time.time()
N = 1500
import sys
MENUS = sys.argv[1:] or ["L abc40 ns35", "L abc40 ns35 shell"]   # read before calib_break resets sys.argv
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")


def feats(v, n0, lab):
    a, b = v[:n0], v[n0:]
    z = (a - a.mean()) / a.std()
    da, db = np.diff(a), np.diff(b)
    return dict(lab=lab, n=len(v), frac=n0 / len(v), mean=a.mean(), sd=a.std(), cv=a.std() / max(abs(a.mean()), 1e-9),
                ac1=np.corrcoef(z[1:], z[:-1])[0, 1], ac2=np.corrcoef(z[2:], z[:-2])[0, 1], kurt=kurtosis(z), skew=skew(z),
                dsd=da.std() / a.std(), rjump=(b.mean() - a.mean()) / a.std(), rsd=np.log(b.std() / a.std()),
                rdsd=np.log(db.std() / da.std()), rac1=np.corrcoef(b[1:], b[:-1])[0, 1] - np.corrcoef(a[1:], a[:-1])[0, 1],
                rkurt=kurtosis(b) - kurtosis(a), maxabs=np.abs(z).max())


S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
y = pd.read_csv("data/train_labels.csv").set_index("id").iloc[:, 0]
ids = np.random.default_rng(0).choice(y.index, N, replace=False)
R = pd.DataFrame([feats(g.value.to_numpy(), int((g.period == 0).sum()), y[i]) for i, g in S[S.id.isin(ids)].groupby("id")])
out = {}
for menu in MENUS:
    B = dict(cb.BREAKS[menu])
    F = pd.DataFrame([feats(*cb.gen(B, 7 * 10 ** 6 + s, bool(s % 2), True), s % 2) for s in range(N)])
    X = pd.concat([R.assign(fake=0), F.assign(fake=1)], ignore_index=True)
    X = X[X.ac1 < .9]
    print(f"\n{menu}: real {int((X.fake == 0).sum())}  fake {int(X.fake.sum())}")
    for lab in (0, 1):
        Z = X[X.lab == lab]
        cols = [c for c in Z if c not in ("lab", "fake")]
        p = cross_val_predict(HistGradientBoostingClassifier(max_iter=150), Z[cols], Z.fake, cv=5, method="predict_proba")[:, 1]
        alone = {c: abs(roc_auc_score(Z.fake, Z[c].fillna(0)) - .5) + .5 for c in cols}
        top = sorted(alone.items(), key=lambda kv: -kv[1])[:5]
        print(f"  label {lab}: real-vs-fake AUC {roc_auc_score(Z.fake, p):.3f}   top single features", [(c, round(a, 3)) for c, a in top])
print(f"\n{time.time() - t0:.1f}s")
