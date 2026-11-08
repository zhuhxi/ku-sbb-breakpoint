"""Fold-clean L2 logistic regression on GPU (Adam, full batch), standardised features. probe(X, y, l2) -> OOF, test."""
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold
def fit_pred(Xtr, ytr, Xev, l2, steps=300, dev="cuda"):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-4
    A = torch.tensor((Xtr - mu) / sd, device=dev); Y = torch.tensor(ytr, device=dev, dtype=torch.float32)
    w = torch.zeros(A.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True); opt = torch.optim.Adam([w, b], 1e-2)
    for _ in range(steps):
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, Y) + l2 * (w ** 2).sum(); loss.backward(); opt.step()
    with torch.no_grad(): return [(torch.tensor((E - mu) / sd, device=dev) @ w + b).cpu().numpy() for E in Xev]
def probe(X, y, Xt, l2):
    n = len(y); o = np.zeros(n); te = np.zeros(len(Xt))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y):
        pv, pt = fit_pred(X[tr], y[tr], [X[va], Xt], l2); o[va] = pv; te += pt / 5
    return o, te
