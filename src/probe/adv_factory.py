"""Train-vs-test adversarial AUC of selected factory columns (drift check before submitting).

    python probe/adv_factory.py st_RES10 st_ARES10 fe ar2 cost --pat "^fe_|^st_|_(ab|ba)_|_gain$"

Also reports the AUC with the rescaled family removed (its share differs on purpose: 1.7% train vs 18% test),
and the most separating columns.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

ap = argparse.ArgumentParser()
ap.add_argument("fams", nargs="+")
ap.add_argument("--pat", default=None)
args = ap.parse_args()

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
N = pd.concat([pd.read_parquet(f"outputs/factory/{f}.parquet") for f in args.fams], axis=1)
if args.pat:
    N = N.loc[:, N.columns.str.contains(args.pat, regex=True)]
N = N.loc[:, ~N.columns.duplicated()].replace([np.inf, -np.inf], np.nan)
resc = pd.concat([pd.read_parquet("outputs/magic_b_train.parquet")["mb_rescaled"],
                  pd.read_parquet("outputs/magic_b_test.parquet")["mb_rescaled"]]).fillna(0)
ids = list(lab.index) + list(sub["id"])
X, r = N.reindex(ids), resc.reindex(ids).to_numpy() == 1
z = np.r_[np.zeros(len(lab)), np.ones(len(sub))]
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, n_jobs=30)
p = cross_val_predict(clf, X, z, cv=4, method="predict_proba")[:, 1]
print(f"{X.shape[1]} cols  adversarial AUC all {roc_auc_score(z, p):.3f}  non-rescaled only {roc_auc_score(z[~r], p[~r]):.3f}")
pn = cross_val_predict(clf, X[~r], z[~r], cv=4, method="predict_proba")[:, 1]
print(f"refit without rescaled rows: {roc_auc_score(z[~r], pn):.3f}")
imp = pd.Series(clf.fit(X[~r], z[~r]).booster_.feature_importance("gain"), X.columns)
print("top separating (non-rescaled):", imp.nlargest(10).round(0).to_dict())
