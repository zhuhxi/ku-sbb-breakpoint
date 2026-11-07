"""Stacking level 0: diverse first-level models with out-of-fold and fold-averaged test predictions.

    python stack/level0.py

Same folds as train.py / ensemble.py / lgb_inter.py: StratifiedKFold(5, shuffle, 42) over the
train_labels.csv order. Adds:
  * single-block LightGBM specialists (level, scale, dependence, shape, boundary windows, bits),
    each also given the segment lengths n0, n1;
  * ExtraTrees on the 242 features (random splits: a different inductive bias);
  * L2 logistic regression on the 242 features.
Writes outputs/stack/level0_new.npz with oof (models x n_train), test (models x n_test), names.
"""
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import LGB_PARAMS  # noqa: E402

BLOCKS = {
    "level": r"^(d_(mean|median|tmean|q05|q95|mean_near|mean_far)$|t_stat|t_p|nl_t_p|mw_p|nl_mw_p$|shift_vs_drift|persist$|persist_m"
             r"|w_dmean|r_dmed|r_zmed|cusum|win\d+_m_|rank_dm|rank_mw|rmean_|bw\d+_dm|stdres_dm)",
    "scale": r"^(lr_|lev_p|nl_lev_p|ansari_p|nl_ansari_p|nl_dlev_p|w_lrstd|r_lrmad|ar_lrres|win\d+_s_|rstd_|diff_|absdiff_|absdev_"
             r"|bw\d+_(lsr|lev)|ewma|stdres_(lsr|lev)|vpersist|vshift|nl_fligner|nl_mood|nl_mw_absdiff|ks_absdev|haar|dks_stat"
             r"|d_dstd|d_dabs|b_dstd|a_dstd|b_dabs|a_dabs|spike_)",
    "dependence": r"(acf|^ar\d|chow|^vr\d|spec|cpg|sacf|^pe\d|zlib|win\d+_r_|slope|zero_cross|vacf|ar_res_kurt)",
    "shape": r"(^ks_stat|^ks_p|^nl_ks_p|^ad$|cvm|wass$|^wass|energy|kurt|skew|glr|lcv_d|cv_b|rank_ks|rank_wass|diff_ks|absdev_ks)",
    "windows": r"^bw\d+_",
}

out = Path("outputs/stack")
out.mkdir(parents=True, exist_ok=True)
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
cl = lambda D: D.replace([np.inf, -np.inf], np.nan).astype(np.float32)
F = cl(pd.read_parquet("outputs/feat_train.parquet").loc[lab.index])
Ft = cl(pd.read_parquet("outputs/feat_test.parquet").loc[sub["id"]])[F.columns]
B = cl(pd.read_parquet("outputs/magic_b_train.parquet").reindex(lab.index))
Bt = cl(pd.read_parquet("outputs/magic_b_test.parquet").reindex(sub["id"]))[B.columns]

sets = {k: [c for c in F.columns if re.search(p, c)] + ["n0", "n1"] for k, p in BLOCKS.items()}
X_of = {k: (F[v], Ft[v]) for k, v in sets.items()}
X_of["bits"] = (pd.concat([B, F[["n0", "n1"]]], axis=1), pd.concat([Bt, Ft[["n0", "n1"]]], axis=1))
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(F, y))


def lgb_fit_predict(Xtr, ytr, Xs, seed):
    a, b = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=seed)
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "colsample_bytree": 0.7}, random_state=seed)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])], eval_metric="auc",
          callbacks=[lgb.early_stopping(300, verbose=False)])
    return [m.predict_proba(X)[:, 1] for X in Xs]


def et_fit_predict(Xtr, ytr, Xs, seed):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    m = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, max_features=0.3, n_jobs=24, random_state=seed)
    m.fit(imp.transform(Xtr), ytr)
    return [m.predict_proba(imp.transform(X))[:, 1] for X in Xs]


def lr_fit_predict(Xtr, ytr, Xs, seed):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=4000))
    m.fit(Xtr, ytr)
    return [m.predict_proba(X)[:, 1] for X in Xs]


jobs = [(f"spec_{k}", lgb_fit_predict, X) for k, X in X_of.items()]
jobs += [("extratrees", et_fit_predict, (F, Ft)), ("logreg", lr_fit_predict, (F, Ft))]
oof = np.zeros((len(jobs), len(y)))
tst = np.zeros((len(jobs), len(Ft)))
for j, (name, fn, (X, Xt)) in enumerate(jobs):
    for f, (tr, va) in enumerate(folds):
        pv, pt = fn(X.iloc[tr], y[tr], [X.iloc[va], Xt], 100 * f + 7)
        oof[j, va] = pv
        tst[j] += pt / len(folds)
    print(f"{name:16s} {X.shape[1]:4d} cols  OOF AUC {roc_auc_score(y, oof[j]):.4f}", flush=True)
np.savez(out / "level0_new.npz", oof=oof, test=tst, names=np.array([n for n, _, _ in jobs]))
print("wrote", out / "level0_new.npz")
