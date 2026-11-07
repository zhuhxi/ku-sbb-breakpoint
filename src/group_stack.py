"""Feature-group stacking (ADIA 2025 winner idea: diversity from different feature sets, research/README.md §7).
One LightGBM per feature group of the combo10gnj columns, same folds as confirm.py; then a second level on the
out-of-fold predictions of the groups + the full combo10gnj LightGBM (outputs/factory/confirm_combo10gnj.npz):
equal rank average, and a logistic regression meta-model fitted by an inner 5-fold CV on the OOF ranks.

  G1 classic : features.py + magic B + harvest
  G2 resid   : st_RES10 / st_ARES10 (statistics on AR residual transforms)
  G3 errcost : fe / fe2 / ar2 / cost (prediction errors, AR(2) and change-point costs)
  G4 gen     : genA / genB / gnoise / gjlr (generator likelihood-ratio features)

    python group_stack.py      # -> outputs/factory/group_stack.npz, prints the comparison
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train import LGB_PARAMS, run_cv

PAT = r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_"     # hpc/gnj.sbatch
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
ld = lambda p, idx: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
keep = json.load(open("outputs/harvest_keep.json"))


def fac(fams, idx):
    N = pd.concat([ld(f"outputs/factory/{f}.parquet", idx) for f in fams], axis=1)
    N = N.loc[:, N.columns.str.contains(PAT, regex=True)]
    return N.loc[:, ~N.columns.duplicated()]


GROUPS = {
    "classic": lambda idx, part: pd.concat([ld(f"outputs/feat_{part}.parquet", idx), ld(f"outputs/magic_b_{part}.parquet", idx),
                                            ld(f"outputs/harvest_{part}.parquet", idx)[keep]], axis=1),
    "resid": lambda idx, part: fac(["st_RES10", "st_ARES10"], idx),
    "errcost": lambda idx, part: fac(["fe", "fe2", "ar2", "cost"], idx),
    "gen": lambda idx, part: fac(["genA", "genB", "gnoise", "gjlr"], idx),
}
full = np.load("outputs/factory/confirm_combo10gnj.npz")
O, T = {"full": full["oof"]}, {"full": full["test"]}
for g, f in GROUPS.items():
    X, Xt = f(lab.index, "train"), f(sub["id"], "test")
    Xt = Xt[X.columns]
    print(f"{g}: {X.shape[1]} columns", flush=True)
    oof, pred, _ = run_cv(X, y, Xt, seeds=range(1), params={**LGB_PARAMS, "n_jobs": 32}, logreg=False, verbose=False)
    O[g], T[g] = oof["lgb"], pred["lgb"]
    print(f"  {g} OOF {roc_auc_score(y, O[g]):.4f}  rank corr with full {np.corrcoef(rankdata(O[g]), rankdata(O['full']))[0, 1]:.3f}", flush=True)
np.savez("outputs/factory/group_stack.npz", **{f"oof_{k}": v for k, v in O.items()}, **{f"test_{k}": v for k, v in T.items()})

R = np.column_stack([rankdata(O[k]) / len(y) for k in O])
base = R[:, 0]
rng = np.random.default_rng(0)
B = [rng.integers(0, len(y), len(y)) for _ in range(500)]


def report(name, o):
    d = [roc_auc_score(y[i], o[i]) - roc_auc_score(y[i], base[i]) for i in B]
    print(f"{name:34s} OOF {roc_auc_score(y, o):.4f}  gain {roc_auc_score(y, o) - roc_auc_score(y, base):+.4f}  "
          f"CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")


report("full + 4 groups, equal rank", R.mean(1))
report("full .6 + groups .1 each", .6 * base + .1 * R[:, 1:].sum(1))
meta = np.zeros(len(y))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=7).split(R, y):
    meta[va] = LogisticRegression(C=1.0).fit(R[tr], y[tr]).predict_proba(R[va])[:, 1]
report("logistic meta (inner 5-fold)", meta)
print("meta coefficients (all data, for reading):", dict(zip(O, np.round(LogisticRegression(C=1.0).fit(R, y).coef_[0], 2))))
