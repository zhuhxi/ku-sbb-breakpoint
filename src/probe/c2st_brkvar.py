"""Break-stat realism of generator variants: real vs variant fakes on g_ cols (genA/genB/gnoise/gjlr), per label, 3 subsample seeds.
   python probe/c2st_brkvar.py v0 vcp20 ..."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
fams = ["genA", "genB", "gnoise", "gjlr"]; g = lambda d: d.loc[:, d.columns.str.match(r"^g_")]
RE = pd.concat([g(pd.read_parquet(f"outputs/factory/{f}.parquet")) for f in fams], axis=1); RE.index = RE.index.astype(str); RE = RE.loc[:, ~RE.columns.duplicated()]
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id").label
def auc(R, F, sd):
    n = min(len(R), len(F)); X = pd.concat([R.sample(n, random_state=sd), F.sample(n, random_state=sd)]).replace([np.inf, -np.inf], np.nan); t = np.r_[np.ones(n), np.zeros(n)]; o = np.zeros(2 * n)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, t):
        o[va] = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, min_child_samples=50, colsample_bytree=.7, verbose=-1, n_jobs=16).fit(X.iloc[tr], t[tr]).predict_proba(X.iloc[va])[:, 1]
    return A(t, o)
for v in sys.argv[1:]:
    F = pd.concat([g(pd.read_parquet(f"outputs/aug/var/var_{v}_{f}.parquet")) for f in fams], axis=1); F.index = F.index.astype(str); F = F.loc[:, ~F.columns.duplicated()][RE.columns]
    M = pd.read_parquet(f"outputs/aug/var/meta_{v}.parquet"); M.index = M.index.astype(str)
    r = {lab: np.mean([auc(RE.loc[L.index[L == lab]], F.loc[F.index.intersection(M.index[M.label == lab])], sd) for sd in range(3)]) for lab in (0, 1)}
    print(f"{v:10s} nb {r[0]:.4f}  brk {r[1]:.4f}  mean {(r[0] + r[1]) / 2:.4f}", flush=True)
