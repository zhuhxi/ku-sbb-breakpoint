"""Raw-sequence model: dilated residual 1D CNN, pooled separately over before/after segments.

    CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/deeplearning/bin/python cnn.py

Uses the same StratifiedKFold(5, shuffle, 42) split as train.py, writes outputs/oof_cnn.npy
(train-label order) and outputs/pred_cnn.npy (sample-submission order) for blending.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

L = 1024


def load(path, ids):
    df = pd.read_parquet(path).sort_values(["id", "time"], kind="stable")
    X = np.zeros((len(ids), 2, L), np.float32)  # value channel, period channel (+1 before, -1 after, 0 pad)
    pos = {k: i for i, k in enumerate(ids)}
    for sid, g in df.groupby("id", sort=False):
        v, p = g["value"].to_numpy(np.float64), g["period"].to_numpy()
        x0 = v[p == 0]
        med = np.median(x0)
        mad = 1.4826 * np.median(np.abs(x0 - med)) + 1e-8
        i = pos[sid]
        X[i, 0, : len(v)] = np.clip((v - med) / mad, -20, 20)
        X[i, 1, : len(v)] = np.where(p == 0, 1.0, -1.0)
    return X


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d)
        self.c2 = nn.Conv1d(c, c, 1)
        self.n = nn.BatchNorm1d(c)

    def forward(self, x):
        return x + self.c2(F.gelu(self.n(self.c1(x))))


class Net(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        # Inputs: z, clipped z, |dz|, segment masks.
        self.inp = nn.Conv1d(5, c, 5, padding=2)
        self.body = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)])
        self.head = nn.Sequential(nn.Linear(8 * c, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def pool(self, h, m):
        n = m.sum(-1).clamp(min=1)  # (B, 1)
        mu = (h * m).sum(-1) / n
        sd = (((h - mu[..., None]) ** 2 * m).sum(-1) / n).clamp(min=1e-6).sqrt()
        return mu, sd

    def forward(self, x):
        z, per = x[:, :1], x[:, 1:2]
        mb, ma = (per > 0).float(), (per < 0).float()
        dz = torch.zeros_like(z)
        dz[..., 1:] = (z[..., 1:] - z[..., :-1]).abs() * ma[..., 1:].add(mb[..., 1:]).clamp(max=1)
        h = self.body(self.inp(torch.cat([z, z.clamp(-4, 4), dz.clamp(max=8), mb, ma], 1)))
        mu_b, sd_b = self.pool(h, mb)
        mu_a, sd_a = self.pool(h, ma)
        e = torch.cat([mu_a - mu_b, (mu_a - mu_b).abs(), torch.log(sd_a / sd_b), mu_b, mu_a, sd_b, sd_a, (mu_a - mu_b) / sd_b], 1)
        return self.head(e).squeeze(-1)


def augment(xb):
    # Sign flip; time reversal (before/after swap keeps the label: a lasting change reversed is still one).
    s = torch.where(torch.rand(len(xb), 1, 1, device=xb.device) < 0.5, -1.0, 1.0)
    xb = xb.clone()
    xb[:, :1] *= s
    rev = torch.rand(len(xb), device=xb.device) < 0.5
    n = (xb[:, 1] != 0).sum(-1, keepdim=True)
    t = torch.arange(xb.shape[-1], device=xb.device)[None]
    idx = torch.where((t < n) & rev[:, None], n - 1 - t, t)
    xb = torch.gather(xb, 2, idx[:, None].expand(-1, 2, -1))
    xb[:, 1] *= torch.where(rev, -1.0, 1.0)[:, None]
    return xb


@torch.no_grad()
def predict(model, X, dev, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    dev = "cuda"
    torch.manual_seed(0)

    labels = pd.read_csv(data / "train_labels.csv", dtype={"id": str})
    sub = pd.read_csv(data / "sample_submission.csv", dtype={"id": str})
    cache = out / "cnn_arrays.npz"
    if cache.exists():
        z = np.load(cache)
        Xtr, Xte = z["tr"], z["te"]
    else:
        Xtr = load(data / "train_series.parquet", labels["id"].tolist())
        Xte = load(data / "test_series.parquet", sub["id"].tolist())
        np.savez(cache, tr=Xtr, te=Xte)
    y = labels["label"].to_numpy().astype(np.float32)

    oof, pred = np.zeros(len(y)), np.zeros(len(Xte))
    for k, (tr, va) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=42).split(Xtr, y)):
        rng = np.random.default_rng(k)
        tr = rng.permutation(tr)
        n_es = len(tr) // 10  # inner split for epoch selection, never the scored fold
        es, tr = tr[:n_es], tr[n_es:]
        model = Net().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        steps = args.epochs * (len(tr) // 128)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
        Xg, yg = torch.from_numpy(Xtr[tr]).to(dev), torch.from_numpy(y[tr]).to(dev)
        best, best_state = -1, None
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(tr), device=dev)
            for i in range(0, len(tr) - 127, 128):
                b = perm[i:i + 128]
                loss = F.binary_cross_entropy_with_logits(model(augment(Xg[b])), yg[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            auc = roc_auc_score(y[es], predict(model, Xtr[es], dev))
            if auc > best:
                best, best_state = auc, {k_: v.clone() for k_, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        oof[va] = predict(model, Xtr[va], dev)
        pred += predict(model, Xte, dev) / args.folds
        print(f"fold {k}: inner-best {best:.4f}  fold AUC {roc_auc_score(y[va], oof[va]):.4f}", flush=True)
    print(f"CNN OOF AUC {roc_auc_score(y, oof):.4f}")
    np.save(out / "oof_cnn.npy", oof)
    np.save(out / "pred_cnn.npy", pred)


if __name__ == "__main__":
    main()
