"""Generator-aware features: what kind of series is this (structure + noise from the two-layer MLE fit), and how
much better do two segment-wise fits of THAT SAME structure explain the data than one whole-series fit
(a likelihood-ratio break test that already knows how this kind of series behaves without a break).

    PYTHONPATH=. python probe/gen_feats.py --n 2000      # small test: baseline top-60 vs + generator features
"""
import argparse
import importlib.util
import sys
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
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--jobs", type=int, default=30)
args = ap.parse_args()
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("cl", "probe/close_loop.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)
mle = cl.mle
STRUCT = ["WN", "AR1", "AR2", "MA1", "ARMA11", "RW", "SIN+WN", "SIN+AR1"]
K = {"WN": 2, "AR1": 3, "AR2": 4, "MA1": 3, "ARMA11": 4, "RW": 2, "SIN+WN": 5, "SIN+AR1": 6}
NOISE = ["normal", "t", "bimodal"]


def ll_of(res, m, n):
    return -(res[m]["bic"] - K[m] * np.log(n - mle.M)) / 2


def gen_features(v, n0):
    n = len(v)
    _, win, rw = mle.fit("w", v)
    _, _, rb = mle.fit("b", v[:n0])
    _, _, ra = mle.fit("a", v[n0:])
    nz, pn = cl.fit_noise(rw[win]["e"])
    f = {"g_struct": STRUCT.index(win), "g_noise": NOISE.index(nz), "g_df": pn.get("df", np.nan),
         "g_bigap": pn["a"] / pn["s"] if nz == "bimodal" else 0.0}
    p = rw[win]
    f["g_phi1"] = p["phi"][0] if "phi" in p else (p["q"][0] if win == "ARMA11" else 0.0)
    f["g_sinamp"] = np.hypot(p["b"][1], p["b"][2]) / p["sigma"] if "b" in p else 0.0
    lr = lambda m: (ll_of(rb, m, n0) + ll_of(ra, m, n - n0) - ll_of(rw, m, n)) / np.sqrt(n)
    f["g_lr_win"] = lr(win)                                 # LR break test under the series' own structure
    for m in ("WN", "AR1", "RW"):                           # reference structures
        f[f"g_lr_{m}"] = lr(m)
    f["g_lr_win_over_wn"] = f["g_lr_win"] - f["g_lr_WN"]
    eb, ea = rw[win]["e"][: max(n0 - mle.M, 5)], rw[win]["e"][max(n0 - mle.M, 5):]
    if nz == "t":                                           # noise-aware scale change: t-likelihood with fixed df
        df = pn["df"]
        llj = stats.t.logpdf(np.r_[eb, ea], df, 0, pn["s"]).sum()
        lls = sum(stats.t.logpdf(e, df, 0, stats.t.fit(e, f0=df, floc=0)[2]).sum() for e in (eb, ea))
        f["g_scale_lr"] = (lls - llj) / np.sqrt(n)
    else:
        f["g_scale_lr"] = 0.5 * (len(eb) * np.log(np.r_[eb, ea].var() / eb.var()) + len(ea) * np.log(np.r_[eb, ea].var() / ea.var())) / np.sqrt(n)
    return f


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    rng = np.random.default_rng(0)
    ids = np.r_[rng.choice(lab.index[lab == 0], args.n // 2, False), rng.choice(lab.index[lab == 1], args.n // 2, False)]
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(ids))].sort_values(["id", "time"])
    items = [(k, g["value"].to_numpy(), int((g["period"] == 0).sum())) for k, g in S.groupby("id")]
    rows = Parallel(n_jobs=args.jobs)(delayed(gen_features)(v, n0) for _, v, n0 in items)
    Gf = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    Gf.to_parquet(f"outputs/probe/gen_feats_{args.n}.parquet")
    print(f"generator features for {len(Gf)} series in {time.time() - t0:.1f}s")

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
    o0, o1 = cv(X), cv(pd.concat([X, Gf], axis=1))
    print(f"top-60 features            AUC {roc_auc_score(y, o0):.4f}")
    print(f"top-60 + generator features AUC {roc_auc_score(y, o1):.4f}")
    neg = y == 0
    thr0, thr1 = np.quantile(o0[~neg], 0.5), np.quantile(o1[~neg], 0.5)
    print(f"false alarms at 50% recall: {np.mean(o0[neg] > thr0):.3f} -> {np.mean(o1[neg] > thr1):.3f}")
    st = Gf["g_struct"].map(dict(enumerate(STRUCT)))
    print("AUC by structure (top-60 -> +gen):")
    for s in STRUCT:
        m = (st == s).to_numpy()
        if m.sum() > 60 and 0 < y[m].mean() < 1:
            print(f"  {s:8s} n={m.sum():4d}  {roc_auc_score(y[m], o0[m]):.3f} -> {roc_auc_score(y[m], o1[m]):.3f}")
    print(f"total {time.time() - t0:.1f}s")
