"""Realism check: simulate from fold-0 training donors, build features, real-vs-sim adversarial AUC.

    python sim/adv_check.py [n] [knobs-json]

Prints the adversarial AUC overall and for no-break / break rows separately (real label vs sim label),
plus the most separating single features.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features  # noqa: E402
from simlib import KNOBS, make_donors, simulate  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
knobs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else KNOBS
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
tr, _ = list(StratifiedKFold(5, shuffle=True, random_state=42).split(lab, lab))[0]
rows = lab.index[tr]
d = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
dd = d[d["id"].isin(set(rows[lab.loc[rows].to_numpy() == 0]))]
donors = make_donors([(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in dd.groupby("id", sort=False)])
parts = Parallel(n_jobs=28)(delayed(simulate)(donors, n // 28, knobs, seed=777 + s, prefix="adv") for s in range(28))
S = build_features(pd.concat([p[0] for p in parts], ignore_index=True))
S["label"] = pd.concat([p[1] for p in parts]).loc[S.index]
F = pd.read_parquet("outputs/feat_train.parquet").loc[rows]
F["label"] = lab.loc[rows]
cols = [c for c in F.columns if c != "label" and "zlib" not in c]   # zlib = float bit trace, not simulable
m = len(S)
F = F.sample(min(m, len(F)), random_state=0)
Z = pd.concat([F, S]).replace([np.inf, -np.inf], np.nan)
z = np.r_[np.zeros(len(F)), np.ones(len(S))]
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=300, verbose=-1, n_jobs=24), Z[cols], z, cv=3, method="predict_proba")[:, 1]
print(f"adversarial AUC real vs sim: all {roc_auc_score(z, p):.3f}", end="")
for L in (0, 1):
    k = Z["label"].to_numpy() == L
    print(f"  label {L}: {roc_auc_score(z[k], p[k]):.3f}", end="")
print()
imp = pd.Series(lgb.LGBMClassifier(n_estimators=300, verbose=-1, n_jobs=24).fit(Z[cols], z).booster_.feature_importance("gain"), cols)
print("top adversarial features:", imp.nlargest(10).round(0).to_dict())
for c in ["ad", "cusum_max", "nl_t_p", "acf_port", "pe5_b"]:
    print(f"  {c:10s} real {F.groupby('label')[c].median().round(3).tolist()}  sim {S.groupby('label')[c].median().round(3).tolist()}")
