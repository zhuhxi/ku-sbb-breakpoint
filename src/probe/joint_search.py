"""Tune the break menu's parameters jointly with the real-vs-fake classifier as the loss (a simple random-search
version of adversarial variational optimization, Louppe et al. 2019). Starts from menu "L abc40 ns35"; every
configuration uses the same real series and the same generator seeds. Results appended to
outputs/probe/joint_search.csv.

    python probe/joint_search.py --k 10 [--seed 0] [--around ROW] [--width 0.5]
"""
import argparse
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--around", type=int, default=-1)      # row of joint_search.csv to search around (-1: wide ranges)
ap.add_argument("--width", type=float, default=0.5)    # fraction of the wide range used around that row
args = ap.parse_args()
sys.argv = [sys.argv[0], "--menu", "L abc40 ns35", "--only-break", "1"]
import importlib.util
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

s = importlib.util.spec_from_file_location("bc", "probe/break_c2st.py")
bc = importlib.util.module_from_spec(s); s.loader.exec_module(bc)
base = bc.cb.BREAKS["L abc40 ns35"]
RANGES = dict(q_level=(.25, .6), q_spread=(.2, .5), q_dep=(.1, .35), small=(.6, .85), mid=(.05, .2),
              noise_switch=(.15, .5), drift_p=(.03, .1), drift_hi=(.5, 2.0))
CURRENT = dict(q_level=.4, q_spread=.35, q_dep=.2, small=.76, mid=.11, noise_switch=.35, drift_p=.08, drift_hi=2.0)
OUT = "outputs/probe/joint_search.csv"


def menu(c):
    grp = lambda a, b, k: list(np.round(np.linspace(a, b, k), 3))
    ns, nm = int(round(c["small"] * 100)), int(round(c["mid"] * 100))
    nb = max(100 - ns - nm, 1)
    return dict(base, switches=dict(level=c["q_level"], spread=c["q_spread"], dep=c["q_dep"]),
                level=(grp(.01, .49, ns) + grp(.8, .9, nm) + grp(1.8, 2.2, nb),),
                noise_switch=c["noise_switch"], drift_p=c["drift_p"], drift_size=(min(.5, c["drift_hi"]), c["drift_hi"]))


lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = set(np.random.default_rng(7).choice(lab.index[lab == 1], 1500, False))
real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
r = np.random.default_rng(1000 + args.seed)
if args.around >= 0:
    c0 = pd.read_csv(OUT).iloc[args.around][list(RANGES)].to_dict()
    lo_hi = {k: (max(a, c0[k] - args.width * (b - a) / 2), min(b, c0[k] + args.width * (b - a) / 2)) for k, (a, b) in RANGES.items()}
    cands = [c0] + [{k: r.uniform(*lo_hi[k]) for k in RANGES} for _ in range(args.k - 1)]
else:
    cands = [CURRENT] + [{k: r.uniform(*RANGES[k]) for k in RANGES} for _ in range(args.k - 1)]
t0 = time.time()
rows = []
for c in cands:
    M = menu(c)
    fake = Parallel(n_jobs=30)(delayed(bc.cb.gen)(M, 5 * 10 ** 6 + 10 ** 5 + i, True, True) for i in range(1500))
    a, tt = bc.c2st(real, fake)
    rows.append({**{k: round(v, 3) for k, v in c.items()}, "auc": round(a, 4), "tell": tt[0]})
    print(f"AUC {a:.3f}  " + " ".join(f"{k}={v:.2f}" for k, v in c.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
print("\nbest:")
print(R.sort_values("auc").head(3).to_string(index=False))
