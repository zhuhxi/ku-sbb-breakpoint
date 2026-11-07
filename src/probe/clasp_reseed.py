"""Re-check the ClaSP gain with other CV fold seeds (features cached by probe/clasp_feats.py).

    python probe/clasp_reseed.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

src = open("probe/feats_combo_cv.py").read().split("o0 = cv(X)")[0]
ns = {"__name__": "fc"}
exec(src, ns)
A, B, X, y, P = ns["A"], ns["B"], ns["X"], ns["y"], ns["P"]
L = pd.read_parquet("outputs/probe/clasp_4000.parquet").reindex(A.index)
base = pd.concat([X, A, B], axis=1)


def cv(Z, s):
    o = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(Z, y):
        o[va] = lgb.LGBMClassifier(**P, random_state=s).fit(Z.iloc[tr], y[tr]).predict_proba(Z.iloc[va])[:, 1]
    return roc_auc_score(y, o)


g = []
for s in (10, 11, 12, 13):
    a0, a1 = cv(base, s), cv(pd.concat([base, L], axis=1), s)
    g.append(a1 - a0)
    print(f"fold seed {s}: without {a0:.4f}  with ClaSP {a1:.4f}  gain {a1 - a0:+.4f}")
print(f"mean gain {np.mean(g):+.4f}")
