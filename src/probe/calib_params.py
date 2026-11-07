"""Simulation calibration of the correlated-formula parameters. Every no-break series (real or simulated) goes through
the same pipeline: remove a straight line if that is what makes it match, take the simplest formula whose residuals
are pure noise, record its fitted parameters. Real and simulated parameter distributions are then compared.

    python probe/calib_params.py real            # fit all real label-0 series, cache
    python probe/calib_params.py sim NAME        # simulate a candidate menu, compare with real
"""
import importlib.util
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter

warnings.filterwarnings("ignore")
mode, name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
sys.argv = [sys.argv[0], "--n", "1"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
em = load("em", "probe/exact_match.py")
cc = load("cc", "probe/calib_corr.py")
mle = em.mle
ORDER = ["WN", "AR1", "AR2", "MA1", "ARMA11"]


def params(m, p):
    if m in ("AR1", "AR2"):
        return dict(a1=p["phi"][0], a2=p["phi"][1] if m == "AR2" else np.nan)
    if m == "MA1":
        return dict(a1=p["q"][0], a2=np.nan)
    if m == "ARMA11":
        return dict(a1=p["q"][0], a2=p["q"][1])
    return dict(a1=np.nan, a2=np.nan)


def classify(v):
    a1 = np.corrcoef(v[1:], v[:-1])[0, 1]
    if a1 > 0.9:
        return dict(cls="RW-like", a1=np.nan, a2=np.nan, trend=0.0)
    for detr in (False, True):
        t = np.arange(len(v))
        b = np.polyfit(t, v, 1) if detr else np.zeros(2)
        x = v - np.polyval(b, t)
        _, win, res = mle.fit("x", x)
        if win.startswith("SIN"):
            return dict(cls="SIN", a1=np.nan, a2=np.nan, trend=0.0)
        for m in ORDER:
            if em.ljung_p(res[m]["e"]) > 0.05:
                return dict(cls=m, trend=b[0] * len(v) / x.std(), **params(m, res[m]))
    return dict(cls="none", a1=np.nan, a2=np.nan, trend=0.0)


def summary(D):
    out = {"share": D.cls.value_counts(normalize=True).reindex(ORDER + ["SIN", "RW-like", "none"]).fillna(0).round(3).to_dict(),
           "trend share": round(float((D.trend != 0).mean()), 3)}
    for m, cols in [("AR1", ["a1"]), ("AR2", ["a1", "a2"]), ("MA1", ["a1"]), ("ARMA11", ["a1", "a2"])]:
        for c in cols:
            z = D.loc[D.cls == m, c]
            out[f"{m} {c}"] = np.round(np.quantile(z, [.05, .25, .5, .75, .95]), 2).tolist() if len(z) else []
    tr = D.trend[D.trend != 0].abs()
    out["|trend rise|"] = np.round(np.quantile(tr, [.05, .25, .5, .75, .95]), 2).tolist() if len(tr) else []
    return out


MENUS = {   # struct shares, AR1 range, AR2 (phi1, phi2) ranges, local-level step sd range, trend prob, trend rise range
    "A": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.85, .85), ar2=((-1.5, 1.5), (-.8, .8)), ll=(.05, .3), tp=.05, tr=(.5, 2.5)),
    "B": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.7, .7), ar2=((-1, 1), (-.6, .6)), ll=(.05, .3), tp=.05, tr=(.5, 2.5)),
    "C": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.7, .7), ar2=((-1, 1), (-.6, .6)), ll=(.02, .15), tp=.05, tr=(.5, 2.5)),
    "D": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.6, .6), ar2=((-.8, .8), (-.6, .6)), ll=(.01, .08), tp=.08, tr=(.5, 2.5)),
    "E": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.7, .7), ar2=((-.8, .8), (-.6, .6)), ll=(.01, .08), tp=.08, tr=(.5, 2.5)),
    "F": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.7, .7), ar2=((-.8, .8), (-.6, .6)), ll=(.02, .1), tp=.08, tr=(.5, 2.5)),
    "G": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.7, .7), ar2=((-.5, .5), (-.5, .5)), ll=(.02, .1), tp=.08, tr=(.5, 2.5)),
    "H": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.7, .7), ar2="roots", ll=(.02, .1), tp=.08, tr=(.5, 2.5)),
    "I": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.7, .7), ar2=((-.8, .8), (-.5, .5)), ll=(.02, .1), tp=.08, tr=(.5, 2.5)),
    "J": dict(p=dict(WN=.42, SIN=.17, RW=.07, AR1=.17, AR2=.10, LL=.07), ar1=(-.7, .7), ar2=((-.9, .9), (-.5, .5)), ll=(.02, .1), tp=.08, tr=(.5, 2.5)),
}


def gen(M, seed):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    e, t = cc.noise(r, n), np.arange(n)
    k = r.choice(list(M["p"]), p=np.array(list(M["p"].values())) / sum(M["p"].values()))
    if k == "WN":
        x = e
    elif k == "SIN":
        x = r.uniform(0.5, 3) * np.sin(2 * np.pi * t / r.integers(12, 33) + r.uniform(0, 6.3)) + e
    elif k == "RW":
        x = np.cumsum(e)
    elif k == "AR1":
        x = lfilter([1], [1, -r.uniform(*M["ar1"])], e)
    elif k == "AR2":
        while M["ar2"] != "roots":
            p1, p2 = r.uniform(*M["ar2"][0]), r.uniform(*M["ar2"][1])
            if p2 + p1 < 1 and p2 - p1 < 1:
                break
        if M["ar2"] == "roots":                              # two real roots, each U[-.8, .8]
            z1, z2 = r.uniform(-.8, .8, 2)
            p1, p2 = z1 + z2, -z1 * z2
        x = lfilter([1], [1, -p1, -p2], e)
    else:                                                    # local level: random-walk level + noise
        x = np.cumsum(cc.noise(r, n) * r.uniform(*M["ll"])) + e
    if k in ("WN", "AR1", "AR2") and r.random() < M["tp"]:
        x = x + r.choice([-1, 1]) * r.uniform(*M["tr"]) * x.std() * t / n
    return classify(x)


if __name__ == "__main__" and mode == "sim":
    t0 = time.time()
    R = summary(pd.read_parquet("outputs/probe/calib_params_real.parquet"))
    S = summary(pd.DataFrame(Parallel(n_jobs=30)(delayed(gen)(MENUS[name], s) for s in range(4000))))
    for k in R:
        print(f"  {k:14s} real {R[k]}\n  {'':14s} sim  {S[k]}")
    print(f"total {time.time() - t0:.1f}s")

if __name__ == "__main__" and mode == "real":
    t0 = time.time()
    f = "outputs/probe/calib_params_real.parquet"
    if mode == "real":
        lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
        S = pd.read_parquet("data/train_series.parquet", columns=["id", "time", "value"])
        S = S[S.id.isin(set(lab.index[lab == 0]))].sort_values(["id", "time"])
        D = pd.DataFrame(Parallel(n_jobs=30)(delayed(classify)(g.to_numpy()) for _, g in S.groupby("id")["value"]))
        D.to_parquet(f)
        for k, v in summary(D).items():
            print(f"  {k:14s} {v}")
    print(f"total {time.time() - t0:.1f}s")
