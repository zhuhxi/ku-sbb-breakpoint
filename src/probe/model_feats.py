"""Features from the generator findings (breaks change level / spread / dependence; sizes in noise-sd units; strong
spread = x2 or x1/2), tested on top of the current top-60 features with a quick CV on a subset.
One regression per series with the pre-boundary AR(2) formula and a step dummy:
    x_t = c + d 1[t >= n0] + p1 x_{t-1} + p2 x_{t-2} + e_t

    python probe/model_feats.py --n 4000
"""
import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--jobs", type=int, default=30)
args = ap.parse_args()


def gfeats(v, n0):
    n = len(v)
    X = np.column_stack([np.ones(n - 2), (np.arange(2, n) >= n0).astype(float), v[1:-1], v[:-2]])
    b, *_ = np.linalg.lstsq(X, v[2:], rcond=None)
    e = v[2:] - X @ b
    k = n0 - 2
    ea, eb = e[:k], e[k:]
    s0 = ea.std()
    cov = np.linalg.pinv(X.T @ X) * e.var()
    f = {}
    f["g_lvl_noise"] = b[1] / max(1 - b[2] - b[3], 0.05) / s0                  # level shift in noise-sd units
    f["g_lvl_t"] = b[1] / np.sqrt(cov[1, 1])                                    # its t statistic
    mad = lambda z: stats.median_abs_deviation(z, scale="normal") + 1e-9
    f["g_lvl_robust"] = (np.median(eb) - np.median(ea)) / mad(ea)
    lr = np.log(eb.std() / s0)
    f["g_spr_log"] = lr                                                          # spread change, log ratio
    f["g_spr_log_robust"] = np.log(mad(eb) / mad(ea))
    f["g_spr_dist_x2"] = min(abs(lr - np.log(2)), abs(lr + np.log(2)))          # distance to the strong x2 / x1/2
    f["g_spr_F"] = (len(ea) * len(eb) / n) ** 0.5 * lr                            # length-weighted
    ar1 = lambda z: (z[1:] - z[1:].mean()) @ (z[:-1] - z[:-1].mean()) / ((z[:-1] - z[:-1].mean()) @ (z[:-1] - z[:-1].mean()))
    pa, pb = ar1(v[:n0]), ar1(v[n0:])
    f["g_dep"] = pb - pa
    f["g_dep_z"] = (pb - pa) / np.sqrt(max((1 - pa ** 2) / n0 + (1 - pb ** 2) / (n - n0), 1e-6))
    f["g_resid_ac1_after"] = ar1(eb)                                              # pre formula fails after the boundary?
    f["g_chi2"] = f["g_lvl_t"] ** 2 + (lr / np.sqrt(1 / (2 * len(ea)) + 1 / (2 * len(eb)))) ** 2 + f["g_dep_z"] ** 2
    f["g_kurt_diff"] = stats.kurtosis(eb) - stats.kurtosis(ea)
    return f


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    rng = np.random.default_rng(0)
    ids = np.r_[rng.choice(lab.index[lab == 0], args.n // 2, False), rng.choice(lab.index[lab == 1], args.n // 2, False)]
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(ids))].sort_values(["id", "time"])
    keys, rows = zip(*[(k, (g["value"].to_numpy(), int((g["period"] == 0).sum()))) for k, g in S.groupby("id")])
    Gf = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(gfeats)(v, n0) for v, n0 in rows), index=keys)
    Gf.to_parquet(f"outputs/probe/model_feats_{args.n}.parquet")
    print(f"{Gf.shape[1]} generator features for {len(Gf)} series in {time.time() - t0:.1f}s")
    top = pd.read_csv("outputs/factory/perm_combo10.csv", index_col=0).index[:60].tolist()
    srcs = ["outputs/feat_train.parquet", "outputs/magic_b_train.parquet", "outputs/harvest_train.parquet"] + \
        [f"outputs/factory/{f}.parquet" for f in ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")]
    X = pd.concat([pd.read_parquet(s, columns=[c for c in pq.read_schema(s).names if c in top]) for s in srcs], axis=1)
    X = X.loc[:, ~X.columns.duplicated()].reindex(Gf.index)
    y = lab.reindex(Gf.index).to_numpy()
    P = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=30, n_estimators=300,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=args.jobs)

    def cv(Z, seeds=3):
        o = np.zeros(len(y))
        for s in range(seeds):
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(Z, y):
                o[va] += lgb.LGBMClassifier(**P, random_state=s).fit(Z.iloc[tr], y[tr]).predict_proba(Z.iloc[va])[:, 1] / seeds
        return o
    o0, o1, o2 = cv(X), cv(pd.concat([X, Gf], axis=1)), cv(Gf)
    a0, a1 = roc_auc_score(y, o0), roc_auc_score(y, o1)
    bs = [roc_auc_score(y[i], o1[i]) - roc_auc_score(y[i], o0[i]) for i in (np.random.default_rng(b).integers(0, len(y), len(y)) for b in range(300))]
    print(f"generator features alone     AUC {roc_auc_score(y, o2):.4f}")
    print(f"top-60 features              AUC {a0:.4f}")
    print(f"top-60 + generator features  AUC {a1:.4f}   gain {a1 - a0:+.4f}  (paired bootstrap 95% CI {np.quantile(bs, .025):+.4f} .. {np.quantile(bs, .975):+.4f})")
    print(f"total {time.time() - t0:.1f}s")
