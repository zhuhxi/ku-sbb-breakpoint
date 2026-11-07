"""Paired 5-seed CV of the harvested set H against the current feature set F + B.

    python cv_harvest.py

Arms share folds (StratifiedKFold(5, shuffle, 42)), inner early-stopping split and seeds:
  F+B · F+B+H (all 85) · F+B+H_cp (change-point cost + ICSS, 37) · F+B+H_rest (catch22 / volatility / z, 48).
Paired bootstrap CI vs F+B, plain and on the test-like mix (rescaled family weighted to its test share).
Saves OOF and fold-averaged test predictions to outputs/harvest_cv.npz.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from train import run_cv

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
ld = lambda p, idx: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
F, Ft = ld("outputs/feat_train.parquet", lab.index), ld("outputs/feat_test.parquet", sub["id"])
B, Bt = ld("outputs/magic_b_train.parquet", lab.index), ld("outputs/magic_b_test.parquet", sub["id"])
H, Ht = ld("outputs/harvest_train.parquet", lab.index), ld("outputs/harvest_test.parquet", sub["id"])
cp = [c for c in H if c.startswith(("H_cp_", "H_icss_"))]
rest = [c for c in H if c not in cp]
FB, FBt = pd.concat([F, B], axis=1), pd.concat([Ft, Bt], axis=1)[list(F) + list(B)]
arms = {"F+B": [], "F+B+H": list(H), "F+B+H_cp": cp, "F+B+H_rest": rest}

flag = B["mb_rescaled"].fillna(0).to_numpy() == 1
pt, pr = Bt["mb_rescaled"].fillna(0).mean(), flag.mean()
w_mix = np.where(flag, pt / pr, (1 - pt) / (1 - pr))

oof, tst = {}, {}
for k, cols in arms.items():
    X, Xt = pd.concat([FB, H[cols]], axis=1), pd.concat([FBt, Ht[cols]], axis=1)
    o, p, imp = run_cv(X, y, Xt, seeds=range(5), logreg=False, verbose=False)
    oof[k], tst[k] = o["lgb"], p["lgb"]
    print(f"{k:12s} {X.shape[1]:4d} cols  OOF AUC {roc_auc_score(y, oof[k]):.4f}  "
          f"test-like {roc_auc_score(y, oof[k], sample_weight=w_mix):.4f}", flush=True)
    if k == "F+B+H":
        imp.sort_values(ascending=False).to_csv("outputs/harvest_importance.csv")
        print("  top gain:", list(imp.sort_values(ascending=False).index[:12]), flush=True)

rng = np.random.default_rng(0)
idx = [rng.integers(0, len(y), len(y)) for _ in range(500)]
for k in list(arms)[1:]:
    d = [roc_auc_score(y[i], oof[k][i]) - roc_auc_score(y[i], oof["F+B"][i]) for i in idx]
    dm = [roc_auc_score(y[i], oof[k][i], sample_weight=w_mix[i]) - roc_auc_score(y[i], oof["F+B"][i], sample_weight=w_mix[i]) for i in idx]
    print(f"{k:12s} vs F+B  delta {np.mean(d):+.4f} CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  "
          f"| test-like CI [{np.percentile(dm, 2.5):+.4f}, {np.percentile(dm, 97.5):+.4f}]")
np.savez("outputs/harvest_cv.npz", names=np.array(list(arms)), oof=np.vstack(list(oof.values())), test=np.vstack(list(tst.values())))
