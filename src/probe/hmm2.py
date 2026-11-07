"""Two-state Gaussian HMM (shared sigma) fitted by Baum-Welch (EM = maximum likelihood)."""
import numpy as np


def fit_hmm2(v, iters=30):
    n = len(v)
    mu = np.array([np.quantile(v, 0.25), np.quantile(v, 0.75)])
    sd = v.std() / 2
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi = np.array([0.5, 0.5])
    ll = -np.inf
    for _ in range(iters):
        B = np.exp(-0.5 * ((v[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) + 1e-300
        al = np.empty((n, 2)); c = np.empty(n)
        al[0] = pi * B[0]; c[0] = al[0].sum(); al[0] /= c[0]
        for t in range(1, n):
            al[t] = (al[t - 1] @ A) * B[t]; c[t] = al[t].sum(); al[t] /= c[t]
        be = np.empty((n, 2)); be[-1] = 1
        for t in range(n - 2, -1, -1):
            be[t] = A @ (B[t + 1] * be[t + 1]) / c[t + 1]
        g = al * be
        xi = (al[:-1, :, None] * A[None] * (B[1:] * be[1:])[:, None, :]) / c[1:, None, None]
        new = np.log(c).sum()
        pi = g[0]
        A = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
        mu = (g * v[:, None]).sum(0) / g.sum(0)
        sd = np.sqrt((g * (v[:, None] - mu) ** 2).sum() / n)
        if new - ll < 1e-6:
            ll = new
            break
        ll = new
    return dict(ll=ll, mu=mu, sd=sd, A=A)
