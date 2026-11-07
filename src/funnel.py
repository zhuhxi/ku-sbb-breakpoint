"""Replicate datatech's selection funnel on KU-SBB and measure each step's real gain vs selection optimism.

    python funnel.py

Stages (as in their write-up, scaled to our feature count):
  S0 all base features -> S1 corr filter (|r|>0.95) -> S2 gain top-K1 -> S3 permutation filter
  -> S4 + pairwise interactions of S3 survivors -> S5 gain top-K2 -> S6 permutation filter (final)

"their way": every selection step runs on ALL rows, then the same 5 folds score each stage.
"nested":    the whole funnel re-runs inside each outer training fold; the outer fold is never seen.
their way - nested = how much of a step's apparent gain is selection optimism.
"""
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

K1, K2, TOP_PAIRS, PERM_EPS = 120, 250, 40, 1e-4
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.3, n_estimators=500, verbose=-1, n_jobs=16)

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.to_numpy()
F = pd.read_parquet("outputs/feat_train.parquet").loc[lab.index]
S = pd.read_parquet("outputs/forensics_train.parquet").loc[lab.index]
glob = ["g_mean", "g_std", "g_cv", "g_absmax_over_std", "b_mean_raw", "b_std_raw", "n", "frac_b", "first_value", "last_value"]
X = pd.concat([F, S[glob].add_prefix("raw_")], axis=1).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def pairs(D, cols):
    out = {}
    for a, b in combinations(cols, 2):
        va, vb = D[a].to_numpy(), D[b].to_numpy()
        out[f"{a}*{b}"], out[f"{a}+{b}"], out[f"{a}-{b}"] = va * vb, va + vb, va - vb
        out[f"{a}/{b}"] = va / (np.abs(vb) + 1e-6) * np.sign(vb + 1e-12)
        out[f"{b}/{a}"] = vb / (np.abs(va) + 1e-6) * np.sign(va + 1e-12)
    return pd.DataFrame(out, index=D.index).astype(np.float32)


def corr_filter(D, thr=0.95):
    c = D.fillna(D.median()).corr().abs().to_numpy()
    keep, cols = [], list(D.columns)
    for i in range(len(cols)):
        if all(c[i, j] <= thr for j in keep):
            keep.append(i)
    return [cols[i] for i in keep]


def gain_top(D, yy, k, seed):
    m = lgb.LGBMClassifier(**P, random_state=seed).fit(D, yy)
    return pd.Series(m.booster_.feature_importance("gain"), D.columns).nlargest(k).index.tolist()


def perm_filter(D, yy, seed):
    a, b = train_test_split(np.arange(len(yy)), test_size=0.25, stratify=yy, random_state=seed)
    m = lgb.LGBMClassifier(**P, random_state=seed).fit(D.iloc[a], yy[a])
    Vb = D.iloc[b].copy()
    base = roc_auc_score(yy[b], m.predict_proba(Vb)[:, 1])
    rng, keep = np.random.default_rng(seed), []
    for c in D.columns:
        orig = Vb[c].to_numpy().copy()
        Vb[c] = rng.permutation(orig)
        if base - roc_auc_score(yy[b], m.predict_proba(Vb)[:, 1]) > PERM_EPS:
            keep.append(c)
        Vb[c] = orig
    return keep


def funnel(rows, seed):
    """Run the full funnel on the given rows only; return the column list after each stage."""
    D, yy = X.iloc[rows], y[rows]
    st = {"S0 all": list(X.columns)}
    st["S1 corr"] = corr_filter(D)
    st["S2 gain"] = gain_top(D[st["S1 corr"]], yy, K1, seed)
    st["S3 perm"] = perm_filter(D[st["S2 gain"]], yy, seed)
    I = pairs(X, st["S3 perm"][:TOP_PAIRS])
    st["S4 +inter"] = st["S3 perm"] + list(I.columns)
    full = pd.concat([X, I], axis=1)
    st["S5 gain"] = gain_top(full.iloc[rows][st["S4 +inter"]], yy, K2, seed + 1)
    st["S6 perm"] = perm_filter(full.iloc[rows][st["S5 gain"]], yy, seed + 1)
    return st, full


def score(full, cols, tr, va, seed):
    m = lgb.LGBMClassifier(**{**P, "n_estimators": 800, "learning_rate": 0.03}, random_state=seed)
    return m.fit(full.iloc[tr][cols], y[tr]).predict_proba(full.iloc[va][cols])[:, 1]


folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
stages = ["S0 all", "S1 corr", "S2 gain", "S3 perm", "S4 +inter", "S5 gain", "S6 perm"]

# their way: select once on all rows, then score every stage on the same folds
st, full = funnel(np.arange(len(y)), 0)
their = {s: np.zeros(len(y)) for s in stages}
for f, (tr, va) in enumerate(folds):
    for s in stages:
        their[s][va] = score(full, st[s], tr, va, f)
sizes = {s: len(st[s]) for s in stages}
print("their way done", sizes, flush=True)

# nested: the funnel never sees the outer validation fold
nested = {s: np.zeros(len(y)) for s in stages}
for f, (tr, va) in enumerate(folds):
    stf, fullf = funnel(tr, 10 + f)
    for s in stages:
        nested[s][va] = score(fullf, stf[s], tr, va, f)
    print(f"nested fold {f} done", {s: len(stf[s]) for s in stages}, flush=True)

rows = []
for s in stages:
    a, b = roc_auc_score(y, their[s]), roc_auc_score(y, nested[s])
    rows.append({"stage": s, "n_feat (their way)": sizes[s], "their way AUC": a, "nested AUC": b, "optimism": a - b})
R = pd.DataFrame(rows).set_index("stage")
R["their step gain"] = R["their way AUC"].diff()
R["real step gain"] = R["nested AUC"].diff()
print(R.round(4).to_string())
R.to_csv("outputs/funnel_results.csv")
