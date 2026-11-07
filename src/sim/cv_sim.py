"""Does simulated data help? Paired CV on the REAL folds; simulated rows only ever enter training.

    python sim/cv_sim.py [--seeds 3] [--weights 1,0.3]

Per fold k (StratifiedKFold(5, shuffle, 42)), features.py columns only:
  real      LightGBM on the real training rows (the reference)
  sim only  trained on outputs/sim/feat_k.parquet alone, scored on the real validation rows (realism)
  real+sim  real training rows + simulated rows with sample weight w
Early stopping always uses a 10% split of the REAL training rows. Also prints a real-vs-sim
adversarial AUC (how distinguishable the simulator is) on fold 0.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--weights", default="1,0.3")
ap.add_argument("--tag", default="")
args = ap.parse_args()
W = [float(w) for w in args.weights.split(",")]

P = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, n_jobs=24)
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.to_numpy()
F = pd.read_parquet("outputs/feat_train.parquet").reindex(lab.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
cols = [c for c in F.columns if "zlib" not in c]   # zlib reads the float bit trace, which the simulator cannot copy
F = F[cols]
resc = pd.read_parquet("outputs/magic_b_train.parquet")["mb_rescaled"].reindex(lab.index).fillna(0).to_numpy() == 1


def fit(Xa, ya, wa, Xb, yb, seed):
    m = lgb.train({**P, "seed": seed}, lgb.Dataset(Xa, ya, weight=wa), 5000, valid_sets=[lgb.Dataset(Xb, yb)],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return m


arms = ["real", "sim only"] + [f"real+sim w={w}" for w in W]
oof = {k: np.zeros(len(y)) for k in arms}
for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(F, y)):
    S = pd.read_parquet(f"outputs/sim/feat{args.tag}_{f}.parquet").replace([np.inf, -np.inf], np.nan)
    Xs, ys = S[cols].astype(np.float32).to_numpy(), S["label"].to_numpy()
    if f == 0:
        Z = np.vstack([F.iloc[tr].to_numpy(), Xs[:len(tr)]])
        za = np.r_[np.zeros(len(tr)), np.ones(min(len(tr), len(Xs)))]
        pa = cross_val_predict(lgb.LGBMClassifier(n_estimators=300, verbose=-1, n_jobs=24), Z, za, cv=3, method="predict_proba")[:, 1]
        print(f"adversarial AUC real vs sim (features.py): {roc_auc_score(za, pa):.3f}", flush=True)
    for s in range(args.seeds):
        a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=100 * f + s)
        Xa, Xb, Xv = F.iloc[a].to_numpy(), F.iloc[b].to_numpy(), F.iloc[va].to_numpy()
        m = fit(Xa, y[a], None, Xb, y[b], 100 * f + s)
        oof["real"][va] += m.predict(Xv, num_iteration=m.best_iteration) / args.seeds
        m = fit(Xs, ys, None, Xb, y[b], 100 * f + s)
        oof["sim only"][va] += m.predict(Xv, num_iteration=m.best_iteration) / args.seeds
        for w in W:
            m = fit(np.vstack([Xa, Xs]), np.r_[y[a], ys], np.r_[np.ones(len(a)), np.full(len(ys), w)], Xb, y[b], 100 * f + s)
            oof[f"real+sim w={w}"][va] += m.predict(Xv, num_iteration=m.best_iteration) / args.seeds
    print(f"fold {f}: " + "  ".join(f"{k} {roc_auc_score(y[va], oof[k][va]):.4f}" for k in arms), flush=True)

rng = np.random.default_rng(0)
idx = [rng.integers(0, len(y), len(y)) for _ in range(500)]
for k in arms:
    d = [roc_auc_score(y[i], oof[k][i]) - roc_auc_score(y[i], oof["real"][i]) for i in idx]
    print(f"{k:16s} OOF AUC {roc_auc_score(y, oof[k]):.4f}  rescaled {roc_auc_score(y[resc], oof[k][resc]):.4f}  "
          f"vs real CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
np.save(f"outputs/sim/oof_cv_sim{args.tag}.npy", np.vstack([oof[k] for k in arms]))
