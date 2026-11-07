"""FeatureBoost-style quick check (OpenFE, Zhang et al. ICML 2023): boost a few trees on the NEW columns only, starting
from the current best model's OOF logit (combo10gnj LGB) as init_score; no retraining of the 888-column model.
5-fold on the first N train series; noise floor = the same with the new columns row-shuffled.

    python probe/fb_quick.py kind N        # per-kind LR features (probe/kind_lr.py), computed on the fly
    python probe/fb_quick.py npy N OOF.npy  # positive control (e.g. the interaction model's OOF, known +0.0027 in blend)
    python probe/fb_quick.py groupby N K   # OpenFE GroupByThenMean on the top-K combo10 features: B - mean(B | decile of A)
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

t0 = time.time()
mode, N = sys.argv[1], int(sys.argv[2])
L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
ids, y = L["id"].to_numpy()[:N], L["label"].to_numpy()[:N]
b = np.clip(np.load("outputs/factory/confirm_combo10gnj.npz")["oof"][:N], 1e-6, 1 - 1e-6)
init = np.log(b / (1 - b))

if mode == "kind":
    sys.path.insert(0, "probe")
    from joblib import Parallel, delayed
    from kind_lr import one
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S["id"].astype(str).isin(set(ids))].sort_values(["id", "time"])
    G = {str(k): (g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)}
    F = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(one)(*G[i]) for i in ids)).astype(float)
elif mode == "npy":   # positive control: another model's OOF as the one new column
    A = np.load(sys.argv[3]); A = (A.T if A.ndim == 2 and A.shape[0] < A.shape[1] else A.reshape(len(A), -1))[:N]
    F = pd.DataFrame(A, columns=[f"oof{i}" for i in range(A.shape[1])])
else:
    K = int(sys.argv[3])
    X = pd.read_parquet("outputs/factory/combo10x_train.parquet")
    X.index = X.index.astype(str)
    imp = pd.read_csv("outputs/factory/imp_combo10.csv", index_col=0).iloc[:, 0]
    top = [c for c in imp.sort_values(ascending=False).index if c in X.columns][:K]
    X = X.loc[ids, top].replace([np.inf, -np.inf], np.nan)
    cols = {}
    for a in top:
        q = pd.qcut(X[a].rank(method="first"), 10, labels=False)
        for c in top:
            if c != a:
                cols[f"{c}|{a}"] = (X[c] - X[c].groupby(q).transform("mean")).to_numpy()
    F = pd.DataFrame(cols)
print(f"{F.shape[1]} new cols, build {time.time() - t0:.1f}s", flush=True)

P = dict(objective="binary", learning_rate=0.05, num_leaves=7, min_data_in_leaf=50, feature_fraction=0.5,
         lambda_l2=5.0, verbose=-1, num_threads=30)


def boost(F):
    o, gain = np.zeros(N), np.zeros(F.shape[1])
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(F, y):
        m = lgb.train(P, lgb.Dataset(F.iloc[tr], y[tr], init_score=init[tr]), 60)
        o[va] = init[va] + m.predict(F.iloc[va], raw_score=True)
        gain += m.feature_importance("gain")
    return o, gain


o, gain = boost(F)
rng = np.random.default_rng(0)
o0, _ = boost(F.sample(frac=1, random_state=1).reset_index(drop=True))
B = [rng.integers(0, N, N) for _ in range(300)]
base = roc_auc_score(y, init)
for nm, oo in (("new", o), ("shuffled", o0)):
    d = [roc_auc_score(y[i], oo[i]) - roc_auc_score(y[i], init[i]) for i in B]
    print(f"{nm:9s} gain {roc_auc_score(y, oo) - base:+.4f}  CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
print(f"base {base:.4f}  {time.time() - t0:.1f}s\ntop columns by gain:")
print(pd.Series(gain, F.columns).sort_values(ascending=False).head(8).round(0).to_string())
