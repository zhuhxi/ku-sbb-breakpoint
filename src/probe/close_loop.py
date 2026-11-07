"""Two-layer re-simulation of no-break series: structure (MLE, probe/mle_noise.py) + noise (MLE on the residuals:
normal / Student-t with fitted df / symmetric bimodal = +-a + N(0, s)). Every series is re-simulated from its own
two-layer fit; a real-vs-fake classifier says whether the loop closes (AUC 0.5 = indistinguishable).

    PYTHONPATH=. python probe/close_loop.py --n 600
"""
import argparse
import importlib.util
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--jobs", type=int, default=30)
args = ap.parse_args()
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("mle", "probe/mle_noise.py")
mle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mle)


def fit_noise(e, full=False):
    """MLE of the innovation distribution; returns (name, params) chosen by BIC (full: also the per-family
    (loglik, n_params, params) table)."""
    e = e - e.mean()
    n = len(e)
    s = e.std()
    out = {"normal": (stats.norm.logpdf(e, 0, s).sum(), 1, dict(s=s))}
    df, _, sc = stats.t.fit(e, floc=0)
    out["t"] = (stats.t.logpdf(e, df, 0, sc).sum(), 2, dict(df=df, s=sc))

    def nll(q):
        a, sg = abs(q[0]), abs(q[1]) + 1e-6
        return -np.log(0.5 * stats.norm.pdf(e, a, sg) + 0.5 * stats.norm.pdf(e, -a, sg) + 1e-300).sum()
    o = minimize(nll, [s, s / 2], method="Nelder-Mead")
    out["bimodal"] = (-o.fun, 2, dict(a=abs(o.x[0]), s=abs(o.x[1]) + 1e-6))
    best = min(out, key=lambda k: -2 * out[k][0] + out[k][1] * np.log(n))
    return (best, out[best][2], out) if full else (best, out[best][2])


def draw(name, p, n, rng):
    if name == "normal":
        return rng.standard_normal(n) * p["s"]
    if name == "t":
        return rng.standard_t(max(p["df"], 1.01), n) * p["s"]
    return rng.choice([-1, 1], n) * p["a"] + rng.standard_normal(n) * p["s"]


def simulate(v, win, p, e):
    """Same structures as mle.simulate, driven by the given innovations e."""
    n = len(v)
    t = np.arange(n)
    if win == "WN":
        return p["mu"] + e
    if win in ("AR1", "AR2"):
        x = np.zeros(n)
        k = len(p["phi"])
        x[:k] = v[:k]
        for i in range(k, n):
            x[i] = p["c"] + sum(ph * x[i - j - 1] for j, ph in enumerate(p["phi"])) + e[i]
        return x
    if win == "MA1":
        return p["mu"] + lfilter([1, p["q"][0]], [1], e)
    if win == "ARMA11":
        return p["mu"] + lfilter([1, p["q"][1]], [1, -p["q"][0]], e)
    if win == "RW":
        return p["x0"] + np.cumsum(np.r_[0.0, p["drift"] + e[1:]])
    s = p["b"][0] + p["b"][1] * np.sin(2 * np.pi * p["f"] * t) + p["b"][2] * np.cos(2 * np.pi * p["f"] * t)
    if win == "SIN+WN":
        return s + e
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = p["c"] + p["phi"][0] * r[i - 1] + e[i]
    return s + r


def one(k, v, seed):
    _, win, res = mle.fit(k, v)
    nz, pn = fit_noise(res[win]["e"])
    rng = np.random.default_rng(seed)
    return k, win, nz, pn, simulate(v, win, res[win], draw(nz, pn, len(v), rng))


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    ids = np.random.default_rng(0).choice(lab.index[lab == 0], args.n, replace=False)
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(ids))].sort_values(["id", "time"])
    G = {k: (g["value"].to_numpy(), g["period"].to_numpy()) for k, g in S.groupby("id")}
    out = Parallel(n_jobs=args.jobs)(delayed(one)(k, v, i) for i, (k, (v, _)) in enumerate(G.items()))
    T = pd.crosstab(pd.Series([o[1] for o in out], name="structure"), pd.Series([o[2] for o in out], name="noise"))
    print(f"two-layer fits of {len(out)} series in {time.time() - t0:.1f}s")
    print(T.to_string())
    tdf = np.array([o[3]["df"] for o in out if o[2] == "t"])
    print("fitted t df quantiles 5/25/50/75/95%:", np.round(np.quantile(tdf, [.05, .25, .5, .75, .95]), 1))
    fake = [pd.DataFrame({"id": k + "_f", "time": np.arange(len(x)), "value": x, "period": G[k][1]}) for k, _, _, _, x in out]
    from features import build_features
    F = build_features(pd.concat([S] + fake), n_jobs=args.jobs)
    yy = np.array([i.endswith("_f") for i in F.index], int)
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=args.jobs)
    pr = cross_val_predict(m, F, yy, cv=5, method="predict_proba")[:, 1]
    imp = pd.Series(m.fit(F, yy).booster_.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    print(f"closing the loop: real vs two-layer re-simulated AUC {roc_auc_score(yy, pr):.3f}  (Gaussian-only was 0.763)")
    print("  tell-tale features:", list(imp.index[:8]))
    print(f"total {time.time() - t0:.1f}s")
