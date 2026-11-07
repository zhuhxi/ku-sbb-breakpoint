"""Rank the real-vs-generated gaps, per class, on a small set of interpretable statistics (one 5-10 s experiment).
Stats per series: lsd (log sd), acf1, kurt of AR(5) residuals, spikes (|z| > 6 per 1000 points), and before/after:
dm (mean change / pre sd), dv (log sd change of AR residuals, AR on pre), da (acf1 change). KS distance real vs fake
(bigger = worse), medians side by side. Also the share of series by acf1 bin.

    python probe/gen_compare.py MENU [MENU ...]
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, kurtosis

menus = sys.argv[1:]
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)
N, P = 2500, 5


def st(x, n0):
    pre = x[:n0]
    A = np.column_stack([np.ones(n0 - P)] + [pre[P - k:n0 - k] for k in range(1, P + 1)])
    b = np.linalg.lstsq(A, pre[P:], rcond=None)[0]
    Af = np.column_stack([np.ones(len(x) - P)] + [x[P - k:len(x) - k] for k in range(1, P + 1)])
    e = x[P:] - Af @ b
    ea, eb = e[:n0 - P], e[n0 - P:]
    m = np.median(ea); s = 1.4826 * np.median(np.abs(ea - m)) + 1e-12
    z = (e - m) / s
    ac = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]
    zz = x - x.mean(); Pw = np.abs(np.fft.rfft(zz))[1:] ** 2; kk = Pw.argmax()
    return dict(pk=Pw[max(kk - 1, 0):kk + 2].sum() / Pw.sum(), lsd=np.log(x.std()), acf1=ac(x), kurt=kurtosis(e), spikes=(np.abs(z) > 6).mean() * 1000,
                dm=(x[n0:].mean() - pre.mean()) / pre.std(), dv=np.log(eb.std() / ea.std()), da=ac(x[n0:]) - ac(pre))


L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
rng = np.random.default_rng(0)
keep = set(L[L == 0].sample(N, random_state=0).index) | set(L[L == 1].sample(N, random_state=0).index)
S = S[S.id.isin(keep)].sort_values(["id", "time"])
R = pd.DataFrame([dict(st(g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())), y=L[k]) for k, g in S.groupby("id", sort=False)])
bins = [-1, -.2, .2, .6, .9, .98, 1]
for mname in menus:
    B = cb.BREAKS[mname]
    out = []
    for y in (0, 1):
        xs = Parallel(n_jobs=30, batch_size=32)(delayed(cb.gen)(B, 70_000 + 2 * s + y, bool(y), True) for s in range(N))
        F = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(st)(x, n0) for x, n0 in xs))
        r = R[R.y == y]
        for c in F.columns:
            a, b = r[c].dropna(), F[c].dropna()
            if c in ("dm", "dv", "da"):
                a, b = a.abs(), b.abs()
            out.append(dict(cls=y, stat=c, ks=round(ks_2samp(a, b).statistic, 3), real_med=round(a.median(), 3), fake_med=round(b.median(), 3),
                            real_q90=round(a.quantile(.9), 3), fake_q90=round(b.quantile(.9), 3)))
        hr, hf = pd.cut(r.acf1, bins).value_counts(normalize=True).sort_index(), pd.cut(F.acf1, bins).value_counts(normalize=True).sort_index()
        print(f"[{mname}] class {y} acf1-bin shares real/fake: " + " ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(hr, hf)))
    O = pd.DataFrame(out)
    print(f"[{mname}] mean KS {O.ks.mean():.4f}")
    print(O.sort_values("ks", ascending=False).to_string(index=False))
