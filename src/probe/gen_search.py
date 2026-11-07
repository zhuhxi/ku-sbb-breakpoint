"""Automatic search of the clean generator's free numbers (probe/ku_gen.py DEFAULT) against real data.
Objective = mean KS distance real vs generated over 8 statistics x 2 classes (as probe/gen_compare.py), fixed seeds
for the fakes so trials are comparable. Workers run in parallel and append to outputs/factory/gen_search.jsonl.

    python probe/gen_search.py cache                    # real-data statistics once -> outputs/factory/real_stats.parquet
    python probe/gen_search.py worker WID SECONDS wide  # random trials around DEFAULT ("narrow": around the current best)
    python probe/gen_search.py best [K]                 # top K trials
    python probe/gen_search.py eval default|old         # objective of DEFAULT / of the old best menu (calib_break X1)
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kurtosis, ks_2samp

sys.path.insert(0, "probe")
from ku_gen import DEFAULT, gen

import os
NF, P5 = int(os.environ.get("GS_NF", 1500)), 5
OUT = os.environ.get("GS_OUT", "outputs/factory/gen_search.jsonl")
SEEDS = [int(v) for v in os.environ.get("GS_SEEDS", "90000").split(",")]   # several fake-seed sets -> less overfitting


def st(x, n0):
    pre = x[:n0]
    A = np.column_stack([np.ones(n0 - P5)] + [pre[P5 - k:n0 - k] for k in range(1, P5 + 1)])
    b = np.linalg.lstsq(A, pre[P5:], rcond=None)[0]
    Af = np.column_stack([np.ones(len(x) - P5)] + [x[P5 - k:len(x) - k] for k in range(1, P5 + 1)])
    e = x[P5:] - Af @ b
    ea, eb = e[:n0 - P5], e[n0 - P5:]
    m = np.median(ea); s = 1.4826 * np.median(np.abs(ea - m)) + 1e-12
    ac = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]
    zz = x - x.mean(); Pw = np.abs(np.fft.rfft(zz))[1:] ** 2; kk = Pw.argmax()
    return dict(pk=Pw[max(kk - 1, 0):kk + 2].sum() / Pw.sum(), lsd=np.log(x.std()), acf1=ac(x), kurt=kurtosis(e),
                spikes=(np.abs((e - m) / s) > 6).mean() * 1000, dm=abs((x[n0:].mean() - pre.mean()) / pre.std()),
                dv=abs(np.log(eb.std() / ea.std())), da=abs(ac(x[n0:]) - ac(pre)))


def objective(P, jobs=5, genf=None):
    R = pd.read_parquet("outputs/factory/real_stats.parquet")
    allks = []
    for sd in SEEDS:
        ks = []
        for y in (0, 1):
            if genf is None:
                xs = [gen(P, sd + 2 * s + y, bool(y))[:2] for s in range(NF)]
            else:
                xs = [genf(sd + 2 * s + y, bool(y)) for s in range(NF)]
            F = pd.DataFrame(Parallel(n_jobs=jobs, batch_size=64)(delayed(st)(x, n0) for x, n0 in xs))
            r = R[R.y == y]
            ks += [ks_2samp(r[c].dropna(), F[c].dropna()).statistic for c in F.columns]
        allks.append(ks)
    ks = np.mean(allks, 0)
    return float(np.mean(ks)), [round(k, 3) for k in ks]


def perturb(base, rng, sig):
    P = {}
    for k, v in base.items():
        if isinstance(v, list):                                       # probability vector
            P[k] = list(rng.dirichlet(np.asarray(v) * (40 / sig ** 2 * .09) + 1e-3))
        elif isinstance(v, tuple):
            lo, hi = v
            f = np.exp(rng.normal(0, sig, 2))
            a, b = lo * f[0], hi * f[1]
            if k in ("decoy_len", "vburst_len"):
                a, b = max(int(round(a)), 2), max(int(round(b)), int(round(a)) + 2)
            if k == "t_df":
                a = max(a, 2.5)
            if k in ("ar1",):
                a, b = float(np.clip(lo + rng.normal(0, sig / 2), -.9, 0)), float(np.clip(hi + rng.normal(0, sig / 2), .2, .95))
            P[k] = (min(a, b), max(a, b) + 1e-6) if k not in ("decoy_len", "vburst_len") else (a, b)
        elif k in ("ar2_p1", "ar2_p2", "sin_scale", "lvl_scale", "dep_big_v"):
            P[k] = float(v * np.exp(rng.normal(0, sig / 2)))
        else:
            P[k] = float(np.clip(v * np.exp(rng.normal(0, sig)), 0, 1))
    return P


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cache":
        L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
        keep = set(L[L == 0].sample(2500, random_state=0).index) | set(L[L == 1].sample(2500, random_state=0).index)
        S = pd.read_parquet("data/train_series.parquet")
        S["id"] = S["id"].astype(str)
        S = S[S.id.isin(keep)].sort_values(["id", "time"])
        rows = Parallel(n_jobs=30)(delayed(lambda k, v, n0: dict(st(v, n0), y=L[k]))(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum()))
                                   for k, g in S.groupby("id", sort=False))
        pd.DataFrame(rows).to_parquet("outputs/factory/real_stats.parquet")
        print("cached", len(rows))
    elif cmd == "worker":
        wid, secs, mode = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
        rng = np.random.default_rng(1000 + wid + int(time.time()) % 1000)
        if mode == "climb":                                           # (1+1) hill climb on the shared best, 3 params at a time
            t0 = time.time()
            while time.time() - t0 < secs:
                rows = [json.loads(l) for l in open(OUT)]
                best = min(rows, key=lambda d: d["obj"])
                base = {k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v) for k, v in best["P"].items()}
                full = perturb(base, rng, .2)
                keys = list(rng.choice(list(DEFAULT), 3, replace=False))
                P = {k: (full[k] if k in keys else base[k]) for k in DEFAULT}
                obj, ks = objective(P)
                if obj < best["obj"]:
                    with open(OUT, "a") as f:
                        f.write(json.dumps(dict(obj=obj, ks=ks, mode="climb", changed=keys,
                                                P={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()})) + "\n")
            sys.exit()
        if mode == "narrow":
            rows = [json.loads(l) for l in open(OUT)]
            base = min(rows, key=lambda d: d["obj"])["P"]
            base = {k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v) for k, v in base.items()}
            sig = .12
        else:
            base, sig = DEFAULT, .3
        t0 = time.time()
        while time.time() - t0 < secs:
            P = perturb(base, rng, sig)
            obj, ks = objective(P)
            with open(OUT, "a") as f:
                f.write(json.dumps(dict(obj=obj, ks=ks, mode=mode, P={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()})) + "\n")
    elif cmd == "best":
        rows = sorted((json.loads(l) for l in open(OUT)), key=lambda d: d["obj"])
        K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        print(f"{len(rows)} trials; best objectives:", [round(d["obj"], 4) for d in rows[:K]])
        b = rows[0]["P"]
        for k in DEFAULT:
            print(f"  {k:13s} default {DEFAULT[k]}  best {np.round(b[k], 3).tolist() if not isinstance(b[k], (int, float)) else round(b[k], 3)}")
    elif cmd == "eval":
        if sys.argv[2] in ("default", "bestjson"):
            Pd = DEFAULT if sys.argv[2] == "default" else dict(DEFAULT, **{k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v)
                                                                              for k, v in json.load(open("outputs/factory/ku_gen_best.json")).items()})
            obj, ks = objective(Pd, jobs=30)
            print("DEFAULT", obj, ks)
            if "--seed-file" in sys.argv:
                with open(OUT, "w") as f:
                    f.write(json.dumps(dict(obj=obj, ks=ks, mode="seed", P={k: (list(v) if isinstance(v, tuple) else v) for k, v in Pd.items()})) + "\n")
        else:
            import importlib.util
            sys.argv = [sys.argv[0]]
            spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
            cb = importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
            B = cb.BREAKS["X1"]
            print("old X1", objective(None, jobs=30, genf=lambda s, y: cb.gen(B, s, y, True)))
