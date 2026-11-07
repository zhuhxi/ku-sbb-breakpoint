"""Reverse-engineering the no-break noise: fit a menu of noise models to each label-0 series by conditional
maximum likelihood (Gaussian innovations), pick the best by BIC, re-simulate every series from its own fit, and
check with a real-vs-fake classifier whether the simulation closes the loop (AUC 0.5 = indistinguishable).

    python probe/mle_noise.py --n 300          # pilot
Writes outputs/probe/mle_fits_{n}.parquet (winner, parameters, BIC of every model).
"""
import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--jobs", type=int, default=30)
args = ap.parse_args()
M = 25                                  # every model is scored on t >= M (same rows -> comparable BIC)


def gll(e):
    s2 = e @ e / len(e)
    return -0.5 * len(e) * (np.log(2 * np.pi * s2) + 1), np.sqrt(s2)


def arma_resid(x, phi, th):
    return lfilter(np.r_[1, -np.asarray(phi)], np.r_[1, np.asarray(th)], x)


def best_sine(v):
    n = len(v)
    t = np.arange(n)
    P = np.abs(np.fft.rfft(v - v.mean())) ** 2
    j = np.argmax(P[3:]) + 3
    best = None
    for f in np.linspace((j - 1) / n, (j + 1) / n, 41):
        X = np.column_stack([np.ones(n), np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
        b, *_ = np.linalg.lstsq(X, v, rcond=None)
        r = v - X @ b
        if best is None or r @ r < best[0]:
            best = (r @ r, f, b)
    return best[1], best[2]


def fit(sid, v):
    n = len(v)
    res = {}

    def put(name, e, k, **par):
        ll, s = gll(e[M:])
        res[name] = dict(bic=-2 * ll + k * np.log(n - M), sigma=s, e=e[M:], **par)

    mu = v.mean()
    x = v - mu
    put("WN", x, 2, mu=mu)
    for p in (1, 2):
        X = np.column_stack([np.ones(n - p)] + [v[p - i:n - i] for i in range(1, p + 1)])
        b, *_ = np.linalg.lstsq(X, v[p:], rcond=None)
        put(f"AR{p}", np.r_[np.zeros(p), v[p:] - X @ b], p + 2, c=b[0], phi=list(b[1:]))
    for name, k, f in (("MA1", 3, lambda q: arma_resid(x, [], [q[0]])),
                       ("ARMA11", 4, lambda q: arma_resid(x, [q[0]], [q[1]]))):
        q0 = [0.3] if name == "MA1" else [0.5, 0.0]
        o = minimize(lambda q: f(q)[M:] @ f(q)[M:], q0, method="L-BFGS-B", bounds=[(-0.99, 0.99)] * len(q0))
        put(name, f(o.x), k, mu=mu, q=list(o.x))
    d = np.r_[0.0, np.diff(v)]
    put("RW", d - d[1:].mean(), 2, drift=d[1:].mean(), x0=v[0])
    fr, bs = best_sine(v)
    t = np.arange(n)
    r = v - (bs[0] + bs[1] * np.sin(2 * np.pi * fr * t) + bs[2] * np.cos(2 * np.pi * fr * t))
    put("SIN+WN", r, 5, f=fr, b=list(bs))
    b, *_ = np.linalg.lstsq(np.column_stack([np.ones(n - 1), r[:-1]]), r[1:], rcond=None)
    put("SIN+AR1", np.r_[0.0, r[1:] - b[0] - b[1] * r[:-1]], 6, f=fr, b=list(bs), phi=[b[1]], c=b[0])
    win = min(res, key=lambda k: res[k]["bic"])
    return sid, win, res


def simulate(v, win, p, rng):
    n = len(v)
    e = rng.standard_normal(n) * p["sigma"]
    t = np.arange(n)
    if win == "WN":
        return p["mu"] + e
    if win in ("AR1", "AR2"):
        x = np.zeros(n)
        x[:len(p["phi"])] = v[:len(p["phi"])]
        for i in range(len(p["phi"]), n):
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


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    ids = np.random.default_rng(0).choice(lab.index[lab == 0], args.n, replace=False)
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(ids))].sort_values(["id", "time"])
    G = {k: (g["value"].to_numpy(), g["period"].to_numpy()) for k, g in S.groupby("id")}
    out = Parallel(n_jobs=args.jobs)(delayed(fit)(k, v) for k, (v, _) in G.items())
    W = pd.Series({k: w for k, w, _ in out})
    print(f"fitted {len(out)} series x 8 models in {time.time() - t0:.1f}s; winner by BIC:")
    for k, c in W.value_counts().items():
        print(f"  {k:8s} {c:4d}  {100 * c / len(W):5.1f}%")
    pd.DataFrame([dict(id=k, win=w, **{f"bic_{m}": r[m]["bic"] for m in r}) for k, w, r in out]).set_index("id") \
        .to_parquet(f"outputs/probe/mle_fits_{args.n}.parquet")

    rng = np.random.default_rng(1)
    fake = []
    for k, w, r in out:
        v, per = G[k]
        fake.append(pd.DataFrame({"id": k + "_f", "time": np.arange(len(v)), "value": simulate(v, w, r[w], rng), "period": per}))
    from features import build_features
    F = build_features(pd.concat([S] + fake), n_jobs=args.jobs)
    yy = np.array([i.endswith("_f") for i in F.index], int)
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=args.jobs)
    pr = cross_val_predict(m, F, yy, cv=5, method="predict_proba")[:, 1]
    imp = pd.Series(m.fit(F, yy).booster_.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    print(f"closing the loop: real vs re-simulated AUC {roc_auc_score(yy, pr):.3f}  (0.5 = indistinguishable)")
    print("  tell-tale features:", list(imp.index[:8]))
    print(f"total {time.time() - t0:.1f}s")
