"""Where does the score leak? Train LightGBM on the real training series (combo10 columns that exist for the fakes),
score generated series of one menu (outputs/aug/X_sim.parquet from probe/sim_series.py + aug_features.py sim), and
report detection per break kind and size: AUC of each kind's positives against all generated negatives.

    python probe/per_kind.py N MENU     # N, MENU as given to probe/sim_series.py
"""
import importlib.util
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

NS, menu = int(sys.argv[1]), sys.argv[2]   # NS: param_quick.py below defines its own N
sys.argv = sys.argv[:1]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cb = load("cb", "probe/calib_break.py")
src = open("probe/param_quick.py").read()
exec(src[src.index("lab = pd.read_csv"):src.index("tr, va = next")])     # real X, y (combo10g columns)
S = pd.read_parquet("outputs/aug/X_sim.parquet")
cols = [c for c in X.columns if c in S.columns]
print(f"columns used {len(cols)} of {X.shape[1]}")
a, b = train_test_split(np.arange(len(y)), test_size=.1, stratify=y, random_state=0)
m = lgb.LGBMClassifier(objective="binary", learning_rate=.05, num_leaves=31, min_child_samples=40, subsample=.8, subsample_freq=1,
                       colsample_bytree=.5, reg_lambda=1., n_estimators=3000, verbose=-1, n_jobs=-1)
m.fit(X[cols].iloc[a], y[a], eval_set=[(X[cols].iloc[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
print("real holdout AUC", round(roc_auc_score(y[b], m.predict_proba(X[cols].iloc[b])[:, 1]), 4))
B = dict(cb.BREAKS[menu])
info = []
for s in range(NS):
    x, n0, d = cb.gen(B, 9 * 10 ** 6 + s, bool(s % 2), "info")
    info.append(dict(id=f"sim{s:06d}", y=s % 2, **{k: d[k] for k in ("structure", "kind", "level", "spread", "dep")}))
I = pd.DataFrame(info).set_index("id")
I["p"] = m.predict_proba(S.reindex(I.index)[cols])[:, 1]
neg = I[I.y == 0]
print("fake AUC overall", round(roc_auc_score(I.y, I.p), 4), "(real holdout above)")
rows = []
for kind, E in I[I.y == 1].groupby("kind"):
    sz = {"level": "level", "spread": "spread", "dep": "dep"}.get(kind)
    r = dict(kind=kind, n=len(E), auc=roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(E))], np.r_[neg.p, E.p]))
    if sz:
        q = pd.qcut(E[sz], 3, labels=["small", "mid", "large"])
        for g, F in E.groupby(q, observed=True):
            r[f"auc_{g}"] = roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(F))], np.r_[neg.p, F.p])
            r[f"size_{g}"] = F[sz].median()
    rows.append(r)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nper structure (all kinds):")
print(pd.Series({st: roc_auc_score(E.y, E.p) for st, E in I.groupby("structure")}).round(3).to_string())
