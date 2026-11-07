"""Seed recovery check: the first random draws of a generator usually fix the series length n and the boundary n0, and the
real data show both exactly. For seeds 0..S-1 and several RNG APIs / formulas, generate (n, n0) and count how many land on a
real (n, n0) pair. If the official generator used these seeds, the hit rate is far above the NULL (n from seed s, n0 from
seed s+1, which breaks the within-seed link while keeping the marginals).
    python probe/seed_scan.py S"""
import random, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
t0 = time.time(); S = int(sys.argv[1])
P = set()
for p in ("train", "test"):
    d = pd.read_parquet(f"data/{p}_series.parquet")
    g = d.groupby("id")
    P |= set(zip(g.size().to_numpy(), g["period"].apply(lambda s: int((s == 0).sum())).to_numpy()))
print(f"{len(P)} distinct real (n, n0) pairs  {time.time() - t0:.0f}s", flush=True)
VARIANTS = {
    "gen_int_unif":  lambda s: (lambda r: (int(r.integers(256, 1025)), r.uniform(.3, .7)))(np.random.default_rng(s)),
    "rs_randint":    lambda s: (lambda r: (int(r.randint(256, 1025)), r.uniform(.3, .7)))(np.random.RandomState(s)),
    "py_random":     lambda s: (lambda r: (r.randint(256, 1024), r.uniform(.3, .7)))(random.Random(s)),
    "gen_frac_first": lambda s: (lambda r: (lambda u: (int(r.integers(256, 1025)), u))(r.uniform(.3, .7)))(np.random.default_rng(s)),
    "gen_n0_int":    lambda s: (lambda r: (lambda n: (n, (r.integers(int(.3 * n), int(.7 * n) + 1) + .5) / n))(int(r.integers(256, 1025))))(np.random.default_rng(s)),
    "rs_n0_int":     lambda s: (lambda r: (lambda n: (n, (r.randint(int(.3 * n), int(.7 * n) + 1) + .5) / n))(int(r.randint(256, 1025))))(np.random.RandomState(s)),
    "torch_like_rs_rand": lambda s: (lambda r: (int(256 + r.rand() * 769), r.rand() * .4 + .3))(np.random.RandomState(s)),
}


def chunk(name, lo, hi):
    f = VARIANTS[name]; out = [f(s) for s in range(lo, hi + 1)]
    hit = {"int": 0, "round": 0, "null": 0}; seeds = []
    for i in range(hi - lo):
        n, u = out[i]; _, u2 = out[i + 1]
        if (n, int(u * n)) in P: hit["int"] += 1; seeds.append(lo + i)
        if (n, int(round(u * n))) in P: hit["round"] += 1
        if (n, int(u2 * n)) in P: hit["null"] += 1
    return hit, seeds[:5]


for name in (sys.argv[2].split(',') if len(sys.argv) > 2 else VARIANTS):
    step = S // 30
    R = Parallel(n_jobs=30)(delayed(chunk)(name, lo, min(lo + step, S)) for lo in range(0, S, step))
    tot = {k: sum(r[0][k] for r in R) for k in ("int", "round", "null")}
    ex = [s for r in R for s in r[1]][:5]
    print(f"{name:14s} seeds 0..{S - 1}: hits int {tot['int']} round {tot['round']} | null {tot['null']}  "
          f"(ratio {tot['int'] / max(tot['null'], 1):.2f})  first hit seeds {ex}  {time.time() - t0:.0f}s", flush=True)
