"""Ten small experiments on how the official generator's code is written (real training data only).
Structure from genA (g_struct). Each experiment prints one compact block.

    python probe/code_guess.py E1 ... E10     # any subset
E1  lengths: n range / uniformity, n0/n range, n0 vs int(frac*n) style
E2  AR1 phi (no-break, pre fit): range and lumpiness (continuous uniform vs a grid / list)
E3  noise tails (WN no-break): implied t df = 4 + 6 / excess kurtosis -> continuous or integer df?
E4  SIN amplitude / noise ratio range (no-break)
E5  shell: base (WN no-break mean) range / flatness; size log2 sd range / flatness
E6  level-jump unit: is |jump| / series-sd or |jump| / noise-sd independent of |phi| (AR1 breaks)?
E7  spread factor (WN breaks): log(sd after / sd before) histogram peaks (list of factors?)
E8  level jump (WN breaks): |mean change| / sd histogram peaks
E9  dependence change (AR1 breaks): phi after - phi before histogram
E10 decimals: are values rounded (float32 / fixed decimals) and is the shell applied last?
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import kstest, kurtosis

STRUCT = ["WN", "AR1", "AR2", "MA1", "ARMA11", "RW", "SIN+WN", "SIN+AR1"]
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
A = pd.read_parquet("outputs/factory/genA.parquet")
A.index = A.index.astype(str)
st = A["g_struct"].reindex(L.index).map(dict(enumerate(STRUCT)))
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
S = S.sort_values(["id", "time"])
G = {k: (g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)}
ids = lambda s, y: [k for k in L.index if st[k] == s and L[k] == y]


def lump(v, lo, hi, nb=80):
    """observed / smoothed bin counts: max ratio and top bin centres (spikes = a grid or a list)"""
    h, e = np.histogram(v, bins=nb, range=(lo, hi))
    sm = np.convolve(h, np.ones(5) / 5, "same") + 1
    r = h / sm
    top = np.argsort(-r)[:4]
    return f"max obs/smooth {r.max():.2f} at " + ", ".join(f"{(e[i] + e[i + 1]) / 2:.3f}" for i in top)


def ar1(x):
    z = x - x.mean()
    return z[1:] @ z[:-1] / (z[:-1] @ z[:-1])


def run(E):
    if E == "E1":
        n = np.array([len(G[k][0]) for k in L.index]); n0 = np.array([G[k][1] for k in L.index])
        f = n0 / n
        print(f"E1 n {n.min()}-{n.max()}, KS vs U(256,1024) p={kstest((n - 256) / 768, 'uniform').pvalue:.3f}; n0/n {f.min():.4f}-{f.max():.4f};"
              f" n0==floor(.3n)? {np.mean(n0 == np.floor(.3 * n)):.4f}; share n0/n<.3 {np.mean(f < .3):.4f} >.7 {np.mean(f > .7):.4f}; n0 {n0.min()}-{n0.max()}")
    if E == "E2":
        ph = np.array([ar1(G[k][0][:G[k][1]]) for k in ids("AR1", 0)])
        print(f"E2 AR1 phi (n={len(ph)}): q01 {np.quantile(ph, .01):.3f} q50 {np.median(ph):.3f} q99 {np.quantile(ph, .99):.3f}; {lump(ph, -.9, .9)}")
    if E == "E3":
        k = np.array([kurtosis(G[i][0]) for i in ids("WN", 0)])
        df = 4 + 6 / k[k > .15]
        print(f"E3 WN excess kurt q10/50/90 {np.quantile(k, [.1, .5, .9]).round(3)}; share kurt>.15 {np.mean(k > .15):.3f}; implied df q10/50/90 {np.quantile(df, [.1, .5, .9]).round(2)}; {lump(df, 4, 20)}")
    if E == "E4":
        out = []
        for i in ids("SIN+WN", 0):
            x = G[i][0]; z = x - x.mean(); P = np.abs(np.fft.rfft(z)) ** 2; kk = P[1:].argmax() + 1
            t = np.arange(len(x)); T = len(x) / kk
            M = np.column_stack([np.ones(len(x)), np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T)])
            c = np.linalg.lstsq(M, x, rcond=None)[0]; r = x - M @ c
            out.append(np.hypot(c[1], c[2]) / r.std())
        o = np.array(out)
        print(f"E4 SIN amp/noise (n={len(o)}): min {o.min():.2f} q05 {np.quantile(o, .05):.2f} q50 {np.median(o):.2f} q95 {np.quantile(o, .95):.2f} max {o.max():.2f}; {lump(o, 0, 4)}")
    if E == "E5":
        m = np.array([G[k][0].mean() for k in ids("WN", 0)]); s = np.array([np.log2(G[k][0].std()) for k in ids("WN", 0)])
        print(f"E5 WN mean {m.min():.3f}..{m.max():.3f} KS vs U(-2,2) p={kstest((m + 2) / 4, 'uniform').pvalue:.3f}; log2 sd {s.min():.3f}..{s.max():.3f} KS vs U(-1,1) p={kstest((s + 1) / 2, 'uniform').pvalue:.3f}")
    if E == "E6":
        rows = []
        for k in ids("AR1", 1):
            x, n0 = G[k]; pre, post = x[:n0], x[n0:]
            ph = ar1(pre); e = pre[1:] - pre.mean() - ph * (pre[:-1] - pre.mean())
            d = abs(post.mean() - pre.mean())
            rows.append((abs(ph), d / pre.std(), d / e.std()))
        R = np.array(rows); big = R[:, 1] > .5
        c1, c2 = np.corrcoef(R[big, 0], R[big, 1])[0, 1], np.corrcoef(R[big, 0], R[big, 2])[0, 1]
        print(f"E6 AR1 breaks with a visible jump (n={big.sum()}): corr(|phi|, jump/series-sd) {c1:+.3f}  corr(|phi|, jump/noise-sd) {c2:+.3f}  (the unit used in the code is the one NOT tied to phi)")
    if E == "E7":
        v = np.array([np.log(G[k][0][G[k][1]:].std() / G[k][0][:G[k][1]].std()) for k in ids("WN", 1)])
        print(f"E7 WN breaks log sd ratio: q05 {np.quantile(v, .05):.3f} q50 {np.median(v):.3f} q95 {np.quantile(v, .95):.3f}; |v|>.1 share {np.mean(np.abs(v) > .1):.3f}; {lump(np.abs(v), 0, 1.2)}")
    if E == "E8":
        v = np.array([abs(G[k][0][G[k][1]:].mean() - G[k][0][:G[k][1]].mean()) / G[k][0][:G[k][1]].std() for k in ids("WN", 1)])
        print(f"E8 WN breaks |mean change|/sd: q50 {np.median(v):.3f} q90 {np.quantile(v, .9):.3f} q99 {np.quantile(v, .99):.3f}; {lump(v, 0, 3)}")
    if E == "E9":
        v = np.array([ar1(G[k][0][G[k][1]:]) - ar1(G[k][0][:G[k][1]]) for k in ids("AR1", 1)])
        print(f"E9 AR1 breaks phi change: q05 {np.quantile(v, .05):.3f} q50 {np.median(v):.3f} q95 {np.quantile(v, .95):.3f}; |dphi|>.15 share {np.mean(np.abs(v) > .15):.3f}; {lump(np.abs(v), 0, 1)}")
    if E == "E10":
        v = np.concatenate([G[k][0] for k in list(L.index[:2000])])
        f32 = np.mean(v.astype(np.float32).astype(np.float64) == v)
        dec = {d: np.mean(np.round(v, d) == v) for d in (4, 6, 8, 10, 12)}
        print(f"E10 exactly float32: {f32:.4f}; exactly rounded to d decimals: " + " ".join(f"{d}:{p:.4f}" for d, p in dec.items()))


for E in sys.argv[1:]:
    run(E)
