"""Equal rank blend of combo10 LGB + every interaction-model arm (all-91k and top-800) for the given tags.

    python blend_v4.py --tags _combo10 _combo10t260 --out outputs/submission_factory_v4.csv

No weights are fitted on OOF (equal weights per arm), so the OOF and CI against v3 are not selection-biased.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--arms", default="12", help="rows of the interaction files to use: 1 = all-91k, 2 = top-800")
ap.add_argument("--out", required=True)
args = ap.parse_args()

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
r = pd.read_parquet("outputs/magic_b_train.parquet")["mb_rescaled"].reindex(lab.index).fillna(0).to_numpy() == 1
pt = pd.read_parquet("outputs/magic_b_test.parquet")["mb_rescaled"].fillna(0).mean()
w = np.where(r, pt / r.mean(), (1 - pt) / (1 - r.mean()))
R = lambda v: rankdata(v) / len(v)

z = np.load("outputs/factory/confirm_combo10.npz")
arms = {"lgb": (z["oof"], z["test"])}
for t in args.tags:
    o, p = np.load(f"outputs/brute/oof_lgb_inter{t}.npy"), np.load(f"outputs/brute/test_lgb_inter{t}.npy")
    for k in args.arms:
        arms[f"{t}:{k}"] = (o[int(k)], p[int(k)])
for k, (o, _) in arms.items():
    print(f"{k:22s} OOF {roc_auc_score(y, o):.4f}")
oof = sum(R(o) for o, _ in arms.values()) / len(arms)
tst = sum(R(p) for _, p in arms.values()) / len(arms)

z1 = np.load("outputs/brute/oof_lgb_inter_combo10.npy")
v3 = (R(z["oof"]) + R(z1[1]) + R(z1[2])) / 3
rng = np.random.default_rng(0)
d = []
for _ in range(500):
    i = rng.integers(0, len(y), len(y))
    d.append(roc_auc_score(y[i], oof[i]) - roc_auc_score(y[i], v3[i]))
print(f"blend of {len(arms)} arms: OOF {roc_auc_score(y, oof):.4f}  test-like {roc_auc_score(y, oof, sample_weight=w):.4f}  "
      f"vs v3 {roc_auc_score(y, oof) - roc_auc_score(y, v3):+.4f} CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
print(f"test rank corr with v3 submission: {np.corrcoef(tst, pd.read_csv('outputs/submission_factory_v3.csv')['break_probability'])[0, 1]:.4f}")
sub["break_probability"] = tst
sub.to_csv(args.out, index=False)
print("wrote", args.out)
