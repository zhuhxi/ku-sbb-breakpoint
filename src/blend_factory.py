"""Rank-blend the factory-era models and write a submission.

    python blend_factory.py --out outputs/submission_factory_v2.csv

Arms (OOF in train-label order, test in sample-submission order): LGB confirm, XGB, CatBoost and the interaction
model (all-91k, top-800) for combo10. Prints each arm, the equal-weight rank blend, a greedy forward blend
(weights in steps of 1/20, chosen on OOF, so slightly optimistic) and the paired CI of the equal blend against
the current submission's OOF (inter_B). The written submission uses the EQUAL blend of the arms that exist.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/submission_factory_v2.csv")
ap.add_argument("--tag", default="_combo10")
args = ap.parse_args()

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
t = args.tag
arms = {}
for name, path in ((f"lgb{t}", f"outputs/factory/confirm{t}.npz"),
                   (f"xgb{t}", f"outputs/factory/div_xgb{t}.npz"), (f"cat{t}", f"outputs/factory/div_cat{t}.npz")):
    if Path(path).exists():
        z = np.load(path)
        arms[name] = (z["oof"], z["test"])
if Path(f"outputs/brute/oof_lgb_inter{t}.npy").exists():
    o, p = np.load(f"outputs/brute/oof_lgb_inter{t}.npy"), np.load(f"outputs/brute/test_lgb_inter{t}.npy")
    arms[f"int91k{t}"], arms[f"int800{t}"] = (o[1], p[1]), (o[2], p[2])

flag = pd.read_parquet("outputs/magic_b_train.parquet")["mb_rescaled"].reindex(lab.index).fillna(0).to_numpy() == 1
pt = pd.read_parquet("outputs/magic_b_test.parquet")["mb_rescaled"].fillna(0).mean()
w_mix = np.where(flag, pt / flag.mean(), (1 - pt) / (1 - flag.mean()))
A = lambda s, w=None: roc_auc_score(y, s, sample_weight=w)
ro = {k: rankdata(v[0]) / len(y) for k, v in arms.items()}
rt = {k: rankdata(v[1]) / len(sub) for k, v in arms.items()}
for k in arms:
    print(f"{k:14s} OOF {A(arms[k][0]):.4f}  test-like {A(arms[k][0], w_mix):.4f}")
eq_o, eq_t = sum(ro.values()) / len(ro), sum(rt.values()) / len(rt)
print(f"{'equal blend':14s} OOF {A(eq_o):.4f}  test-like {A(eq_o, w_mix):.4f}  ({len(arms)} arms)")

wts = {k: 0 for k in arms}
for _ in range(20):
    best = max(arms, key=lambda k: A(sum((wts[j] + (j == k)) * ro[j] for j in arms)))
    wts[best] += 1
print("greedy weights /20:", wts, f"OOF {A(sum(wts[k] * ro[k] for k in arms)):.4f}")

ref = np.load("outputs/brute/oof_lgb_inter_B.npy").mean(0)
rng = np.random.default_rng(0)
d = []
for _ in range(500):
    i = rng.integers(0, len(y), len(y))
    d.append(roc_auc_score(y[i], eq_o[i]) - roc_auc_score(y[i], ref[i]))
print(f"equal blend vs inter_B: {A(eq_o) - A(ref):+.4f}  CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
sub["break_probability"] = eq_t
sub.to_csv(args.out, index=False)
print("wrote", args.out)
