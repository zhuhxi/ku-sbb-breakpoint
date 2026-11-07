"""Full brute-force interaction search, step 2: GPU screening of every pair, then triples.

    CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/deeplearning/bin/python brute/screen.py [--swap]

Score of a candidate = |Spearman corr(candidate, residual)| on screening rows, i.e. how much
it explains what the base model gets wrong. The same pass scores 3 shuffled residuals; the
largest shuffled score over all candidates of a stage is that stage's noise ceiling.
Only candidates above the ceiling are kept (plus the top 3000 per stage for reference).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--swap", action="store_true")
ap.add_argument("--n_tri", type=int, default=100)
args = ap.parse_args()
out = Path("outputs/brute") / ("swap" if args.swap else "main")
z = np.load(out / "prep.npz", allow_pickle=True)
dev = "cuda"
scr = z["screen"]
RAW = torch.from_numpy(z["raw"][scr]).to(dev)
RNK = torch.from_numpy(z["rank"][scr]).to(dev)
names = list(z["names"])
n, d = RAW.shape

g = torch.Generator(device="cpu").manual_seed(0)
r = torch.from_numpy(z["resid"])
Rm = torch.stack([r] + [r[torch.randperm(n, generator=g)] for _ in range(3)], 1).to(dev)
Rm = (Rm - Rm.mean(0)) / Rm.std(0)

EPS = (RAW.abs().median(0).values * 1e-3 + 1e-9)


def pair_ops(i, J):
    a, b = RAW[:, i:i + 1], RAW[:, J]
    ea, eb = EPS[i], EPS[J]
    ra, rb = RNK[:, i:i + 1], RNK[:, J]
    sb = torch.where(b >= 0, 1.0, -1.0)
    sa = torch.where(a >= 0, 1.0, -1.0)
    return {
        "mul": a * b, "div": a / (b.abs() + eb) * sb, "rdiv": b / (a.abs() + ea) * sa,
        "mul_b2": a * b * b, "mul_a2": a * a * b, "logratio": torch.log(a.abs() + ea) - torch.log(b.abs() + eb),
        "r_add": ra + rb, "r_sub": ra - rb, "r_mul": ra * rb, "r_max": torch.maximum(ra, rb),
        "r_min": torch.minimum(ra, rb), "r_absdiff": (ra - rb).abs(),
    }


def tri_ops(i, j, K):
    a, b, c = RAW[:, i:i + 1], RAW[:, j:j + 1], RAW[:, K]
    ra, rb, rc = RNK[:, i:i + 1], RNK[:, j:j + 1], RNK[:, K]
    sc = torch.where(c >= 0, 1.0, -1.0)
    return {"mul3": a * b * c, "muldiv": a * b / (c.abs() + EPS[K]) * sc,
            "r_addmul": (ra + rb) * rc, "r_submul": (ra - rb) * rc}


def score(C):
    """|Spearman| of each column of C (n x k) against the 4 residual columns -> (k, 4)."""
    C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    rk = C.argsort(0).argsort(0).float()
    rk = (rk - rk.mean(0)) / (rk.std(0) + 1e-6)
    return (rk.T @ Rm / (n - 1)).abs()


def run_stage(tag, gen):
    t, recs, null_max, total = time.time(), [], 0.0, 0
    for key, ops in gen:
        for op, C in ops.items():
            s = score(C).cpu().numpy()
            total += len(s)
            null_max = max(null_max, float(s[:, 1:].max()))
            recs.append(pd.DataFrame({"op": op, "key": [key + (k,) for k in range(len(s))], "score": s[:, 0]}))
    df = pd.concat(recs, ignore_index=True)
    print(f"{tag}: {total:,} candidates scored in {time.time() - t:.0f}s; noise ceiling {null_max:.4f}; "
          f"above ceiling {int((df.score > null_max).sum())}; best real {df.score.max():.4f}", flush=True)
    return df, null_max


def pairs_gen():
    for i in range(d - 1):
        J = torch.arange(i + 1, d, device=dev)
        yield (i,), pair_ops(i, J)


pairs, null_p = run_stage("pairs", pairs_gen())
# decode pair keys -> (a, b)
pairs["a"] = pairs.key.map(lambda k: k[0])
pairs["b"] = pairs.key.map(lambda k: k[0] + 1 + k[1])
pairs = pairs.drop(columns="key").sort_values("score", ascending=False)
keep_p = pairs[(pairs.score > null_p)]
top_ref = pairs.head(3000)
pd.concat([keep_p, top_ref]).drop_duplicates(["op", "a", "b"]).assign(
    a_name=lambda x: x.a.map(lambda i: names[i]), b_name=lambda x: x.b.map(lambda i: names[i]),
    passed=lambda x: x.score > null_p).to_parquet(out / "screen_pairs.parquet")

# base features for triples: the ones that appear in the best pairs, in order, up to n_tri
order = []
for a_, b_ in zip(pairs.a, pairs.b):
    for c in (a_, b_):
        if c not in order:
            order.append(c)
    if len(order) >= args.n_tri:
        break
T = order[: args.n_tri]
null_t = None
if len(keep_p) == 0:
    print("no pair beats the noise ceiling -> stop rule: triples skipped")
else:
    Tt = torch.tensor(T, device=dev)

    def tri_gen():
        for x in range(len(T)):
            for y_ in range(x + 1, len(T)):
                K = Tt[y_ + 1:]
                if len(K):
                    yield (T[x], T[y_]), tri_ops(T[x], T[y_], K)

    tri, null_t = run_stage("triples", tri_gen())
    pos = {c: k for k, c in enumerate(T)}
    tri["a"] = tri.key.map(lambda k: k[0])
    tri["b"] = tri.key.map(lambda k: k[1])
    tri["c"] = tri.key.map(lambda k: T[pos[k[1]] + 1 + k[2]])
    tri = tri.drop(columns="key").sort_values("score", ascending=False)
    pd.concat([tri[tri.score > null_t], tri.head(3000)]).drop_duplicates(["op", "a", "b", "c"]).assign(
        passed=lambda x: x.score > null_t).to_parquet(out / "screen_triples.parquet")

json.dump({"pairs_noise_ceiling": null_p, "pairs_above": int(len(keep_p)),
           "triples_noise_ceiling": null_t, "triple_base_features": [names[i] for i in T]},
          open(out / "null_max.json", "w"), indent=1)
print("wrote", out)
