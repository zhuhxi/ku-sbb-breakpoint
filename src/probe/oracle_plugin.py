"""Plug-in oracle: the same likelihood ratio as probe/oracle_ceiling.py, but every nuisance parameter is ESTIMATED from the
pre-boundary part of the series (what a real feature could do): shell a, b = pre mean / sd; AR1 phi by least squares;
noise Gaussian vs unit-variance t (df on a grid) by maximum likelihood on the pre innovations; family = AR1 if |phi| > .1.
Same 40k fakes (seeds 0..N-1) as outputs/factory/oracle.parquet.     python probe/oracle_plugin.py
"""
import sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "probe"); import oracle_ceiling as O


def estimate(x, n0):
    a, b = x[:n0].mean(), x[:n0].std()
    z = (x[:n0] - a) / b
    p1 = float(np.clip(np.dot(z[1:], z[:-1]) / np.dot(z[:-1], z[:-1]), -.95, .95))
    f = z * O.sd_ar(p1); e = f[1:] - p1 * f[:-1]; esd = e.std(); e = e / esd
    best = ("g", 0.0, O.logf(e, "g", 0).sum())
    for df in np.linspace(2.5, 30, 40):
        ll = O.logf(e, "t", df).sum()
        if ll > best[2]: best = ("t", df, ll)
    return dict(fam="AR1" if abs(p1) > .1 else "WN", p1=p1, ty=best[0], df=best[1], a=a, b=b, esd=1.0)


def one(s):
    x, n0, q = O.gen(s, bool(s % 2))
    return O.score(x, n0, estimate(x, n0), s)


if __name__ == "__main__":
    sys.path.insert(0, "probe"); import oracle_plugin as OP
    t0 = time.time()
    D = pd.read_parquet("outputs/factory/oracle.parquet")
    D["plug"] = Parallel(n_jobs=30, batch_size=16)(delayed(OP.one)(s) for s in range(len(D)))
    D.to_parquet("outputs/factory/oracle.parquet")
    print(f"{len(D)} fakes  oracle (true parameters) {roc_auc_score(D.label, D.score):.4f} | plug-in (estimated) {roc_auc_score(D.label, D.plug):.4f}  {time.time() - t0:.0f}s")
    neg = D[D.label == 0]
    for k in ("spread", "level", "dep", "shape"):
        g = pd.concat([neg, D[D.kind == k]]); print(f"  {k:6s} oracle {roc_auc_score(g.label, g.score):.4f}  plug-in {roc_auc_score(g.label, g.plug):.4f}")
    for f in ("WN", "AR1"):
        g = D[D.fam == f]; print(f"  {f:4s} oracle {roc_auc_score(g.label, g.score):.4f}  plug-in {roc_auc_score(g.label, g.plug):.4f}")
