"""Use every brute-force candidate at once: 855,050 pairwise interactions + 350 pool columns.

    CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/deeplearning/bin/python brute/allin.py [--smoke]

The full matrix would be ~82 GB, so features are computed on the GPU per mini-batch from the
350 pool columns and never stored. Model: L2-regularised logistic regression. Per outer fold
(same StratifiedKFold(5, shuffle, 42) as train.py): standardisation stats from training rows,
lambda chosen on an inner 10% split, early stopping on that split, then score the outer fold.
Arms: "base only" (350 columns) and "all-in" (350 + 855,050). OOF saved for blending.
Clip/rank maps use all training rows' features (no labels).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--epochs", type=int, default=12)
args = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.to_numpy().astype(np.float32)
load = lambda p: pd.read_parquet(p).reindex(lab.index).replace([np.inf, -np.inf], np.nan)
F = pd.concat([load("outputs/feat_train.parquet"), load("outputs/raw_train.parquet")], axis=1).to_numpy(np.float64)
lo, hi = np.nanquantile(F, 0.001, axis=0), np.nanquantile(F, 0.999, axis=0)
med = np.nanmedian(F, axis=0)
RAW = np.where(np.isnan(F), med, np.clip(F, lo, hi))
RNK = np.zeros_like(F)
for j in range(F.shape[1]):
    c = F[:, j]
    s = np.sort(c[~np.isnan(c)])
    RNK[:, j] = np.where(np.isnan(c), 0.0, np.searchsorted(s, c) / max(len(s), 1) - 0.5)
RAW, RNK = torch.tensor(RAW, dtype=torch.float32, device=dev), torch.tensor(RNK, dtype=torch.float32, device=dev)
d = RAW.shape[1]
ia, ib = torch.triu_indices(d, d, 1, device=dev)
EPS = RAW.abs().median(0).values * 1e-3 + 1e-9
slog = lambda x: torch.sign(x) * torch.log1p(x.abs())
N_INT = 14 * len(ia)
print(f"pool {d} columns, {len(ia):,} pairs x 14 operators = {N_INT:,} interaction features", flush=True)


def feats(rows, with_inter=True):
    a, b = RAW[rows][:, ia], RAW[rows][:, ib]
    base = torch.cat([slog(RAW[rows]), RNK[rows]], 1)
    if not with_inter:
        return base
    ra, rb = RNK[rows][:, ia], RNK[rows][:, ib]
    ea, eb = EPS[ia], EPS[ib]
    sa, sb = torch.where(a >= 0, 1.0, -1.0), torch.where(b >= 0, 1.0, -1.0)
    raw_ops = [a * b, a / (b.abs() + eb) * sb, b / (a.abs() + ea) * sa, a * b * b, a * a * b,
               torch.log(a.abs() + ea) - torch.log(b.abs() + eb), a + b, a - b]
    rank_ops = [ra + rb, ra - rb, ra * rb, torch.maximum(ra, rb), torch.minimum(ra, rb), (ra - rb).abs()]
    X = torch.cat([base] + [slog(o) for o in raw_ops] + rank_ops, 1)
    return torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def stats(rows, with_inter, bs=256):
    s = s2 = None
    for i in range(0, len(rows), bs):
        X = feats(rows[i:i + bs], with_inter).double()
        s = X.sum(0) if s is None else s + X.sum(0)
        s2 = (X * X).sum(0) if s2 is None else s2 + (X * X).sum(0)
    m = s / len(rows)
    sd = (s2 / len(rows) - m * m).clamp(min=0).sqrt()
    return m.float(), torch.where(sd > 1e-8, sd, torch.ones_like(sd)).float()


@torch.no_grad()
def predict(w, b0, rows, m, sd, with_inter, bs=512):
    out = []
    for i in range(0, len(rows), bs):
        X = (feats(rows[i:i + bs], with_inter) - m) / sd
        out.append(torch.sigmoid(X @ w + b0))
    return torch.cat(out).cpu().numpy()


def train(rows_tr, rows_es, m, sd, lam, with_inter, bs=256):
    p = m.numel()
    w = torch.zeros(p, device=dev, requires_grad=True)
    b0 = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b0], lr=1e-3)
    yt = torch.tensor(y, device=dev)
    best, best_w, bad = -1, None, 0
    for ep in range(args.epochs):
        perm = rows_tr[torch.randperm(len(rows_tr), device=dev)]
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            X = (feats(r, with_inter) - m) / sd
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b0, yt[r]) + lam * (w * w).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = roc_auc_score(y[rows_es.cpu().numpy()], predict(w, b0, rows_es, m, sd, with_inter))
        if auc > best:
            best, best_w, bad = auc, (w.detach().clone(), b0.detach().clone()), 0
        else:
            bad += 1
            if bad >= 3:
                break
    return best, best_w


LAMS = (1e-5, 1e-4, 1e-3, 1e-2, 3e-2)
arms = {"base only": False, "all-in": True}
oof = {k: np.zeros(len(y)) for k in arms}
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y))
if args.smoke:
    folds = folds[:1]
for f, (tr, va) in enumerate(folds):
    a_, b_ = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=f)
    if args.smoke:
        a_ = a_[:4000]
    A, B, V = (torch.tensor(x, device=dev) for x in (a_, b_, va))
    for k, wi in arms.items():
        t = time.time()
        m, sd = stats(A, wi)
        tried = {lam: train(A, B, m, sd, lam, wi) for lam in LAMS}
        lam = max(tried, key=lambda l: tried[l][0])
        w, b0 = tried[lam][1]
        oof[k][va] = predict(w, b0, V, m, sd, wi)
        print(f"fold {f} {k:9s}: lambda {lam:g} (inner AUCs " + ", ".join(f"{l:g}:{tried[l][0]:.4f}" for l in LAMS) +
              f")  outer AUC {roc_auc_score(y[va], oof[k][va]):.4f}  [{time.time() - t:.0f}s, {m.numel():,} features]", flush=True)
        del m, sd, tried
        torch.cuda.empty_cache()
    if args.smoke:
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        raise SystemExit

for k in arms:
    print(f"{k:9s} OOF AUC {roc_auc_score(y, oof[k]):.4f}")
np.save("outputs/brute/oof_allin.npy", np.vstack([oof[k] for k in arms]))
