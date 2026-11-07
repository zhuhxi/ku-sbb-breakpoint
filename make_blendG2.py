"""Builds the designated GPU-route submission blendG2 (Kaggle public 0.86386) from the per-arm prediction files.

blendG2 = rank( 2*rank(mean rank of 26 pair arms) + 1*rank(interaction arm top-800) + 2*rank(mean rank of 8 time-reversal pair arms)
                + 6*rank(mean rank of 8 TabPFN top-K arms) + 6*rank(mean rank of 5 reversed-context TabPFN arms) ),
every vector rounded to 12 decimals before ranking. Weights were fixed before this run (same as blendG), not tuned on the leaderboard.
Arms (see TRAINING_GPU.md): pair arms  src/probe/confirm_pair.py ; time-reversal pair arms  src/probe/confirm_pair_rev.py ;
interaction arm  src/brute/lgb_inter.py ; TabPFN arms  src/probe/tabpfn_top.py (GPU) ; reversed-context TabPFN  tabpfn_top.py REV=.. CTXREV=1.
    python make_blendG2.py --root <project outputs root>
"""
import argparse, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); a = ap.parse_args(); R = a.root
sub = pd.read_csv(f"{R}/data/sample_submission.csv", dtype={"id": str}); r = lambda v: rankdata(np.round(v, 12)) / len(v)
pf = sorted(glob.glob(f"{R}/outputs/factory/confirm_pair[0-9bs]*.npz")); vf = sorted(glob.glob(f"{R}/outputs/factory/confirm_prev*.npz"))
skip = ("tpf30_", "tpf150_", "ne16", "c8e2", "c4e1", "ne1", "rtta", "rctx", "ps25", "psel")
tf = [f for f in sorted(glob.glob(f"{R}/outputs/arms/tpf*_test.npy")) if not any(x in f for x in skip)]
cf = [f for f in sorted(glob.glob(f"{R}/outputs/arms/tpf*rctx*_test.npy")) if "noTTA" not in f]
assert (len(pf), len(vf), len(tf), len(cf)) == (26, 8, 8, 5), (len(pf), len(vf), len(tf), len(cf))
It = np.load(f"{R}/outputs/brute/test_lgb_inter_gnjt162.npy")
pa = r(sum(r(np.load(f)["test"]) for f in pf)); pv = r(sum(r(np.load(f)["test"]) for f in vf)); ta = r(sum(r(np.load(f)) for f in tf)); ca = r(sum(r(np.load(f)) for f in cf))
pd.DataFrame({"id": sub.id, "break_probability": r(2 * pa + r(It[2]) + 2 * pv + 6 * ta + 6 * ca)}).to_csv("submission_blendG2.csv", index=False)
