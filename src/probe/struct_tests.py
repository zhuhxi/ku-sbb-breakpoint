"""Which way is each step of the official generator written? One small data test per structural question
(real training data, structure from genA g_struct). Run any subset; each prints one block.

    python probe/struct_tests.py S1 ... S7
S1 level break: added to the OUTPUT (AR residuals show a spike d at n0, then (1-phi) d)  or  through the AR INTERCEPT
   (residuals shift by D from n0 on, no spike)?  -> median spike/plateau vs median 1/(1-phi), AR1 breaks, phi > .3
S2 spread break on sine series: is the WHOLE output scaled (sine amplitude ratio = noise sd ratio) or only the noise
   (amplitude ratio = 1)?  -> slope of log amp ratio on log noise ratio
S3 dependence break: is the series re-normalised after the AR coefficient changes (variance unchanged) or not
   (log var ratio follows log of the AR variance ratio)?  -> slope
S4 random walk breaks: one big increment at n0 (step on the output) or a shifted mean increment (drift on the steps)?
   -> label AUC of the boundary-increment z and of the drift t
S5 trend layer: is the straight-line trend in both classes (added independently of the break)?
S6 a 4th "shape" mechanism? white-noise breaks with unchanged mean / sd / acf: does the distribution shape still differ
   from negatives?  (official text names only level, spread, dependence)
S7 is the boundary noise RE-DRAWN (new noise family after n0) also in no-break series?  -> kurtosis change after vs
   before in label 0 vs a pseudo split inside the pre part
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, ks_2samp
from sklearn.metrics import roc_auc_score

STRUCT = ["WN", "AR1", "AR2", "MA1", "ARMA11", "RW", "SIN+WN", "SIN+AR1"]
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
A = pd.read_parquet("outputs/factory/genA.parquet")
A.index = A.index.astype(str)
st = A["g_struct"].reindex(L.index).map(dict(enumerate(STRUCT)))
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
S = S.sort_values(["id", "time"])
G = {k: (g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)}
ids = lambda s, y=None: [k for k in L.index if st[k] in (s if isinstance(s, tuple) else (s,)) and (y is None or L[k] == y)]
ac = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]


def ar1fit(v):
    X = np.column_stack([np.ones(len(v) - 1), v[:-1]])
    c, ph = np.linalg.lstsq(X, v[1:], rcond=None)[0]
    return c, ph, v[1:] - c - ph * v[:-1]


def sinefit(x):
    z = x - x.mean(); P = np.abs(np.fft.rfft(z)) ** 2; k = P[1:].argmax() + 1; T = len(x) / k
    return T


def sinamp(x, T):
    t = np.arange(len(x))
    M = np.column_stack([np.ones(len(x)), np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T)])
    c = np.linalg.lstsq(M, x, rcond=None)[0]
    return np.hypot(c[1], c[2]), (x - M @ c).std()


def run(E):
    if E == "S1":
        out = []
        for k in ids("AR1", 1):
            x, n0 = G[k]
            c, ph, e = ar1fit(x[:n0])
            if ph < .3:
                continue
            r = x[n0:] - c - ph * np.r_[x[n0 - 1], x[n0:-1]]
            sd = e.std()
            plateau = r[3:40].mean()
            if abs(plateau) < 4 * sd / np.sqrt(37):
                continue
            out.append((r[0] / plateau, 1 / (1 - ph)))
        o = np.array(out)
        print(f"S1 AR1 breaks phi>.3 with a clear residual shift (n={len(o)}): median spike/plateau {np.median(o[:, 0]):.2f}, "
              f"median 1/(1-phi) {np.median(o[:, 1]):.2f}  (output step -> ~1/(1-phi); through intercept -> ~1); "
              f"share ratio closer to 1/(1-phi) {np.mean(np.abs(o[:, 0] - o[:, 1]) < np.abs(o[:, 0] - 1)):.2f}")
    if E == "S2":
        out = []
        for k in ids("SIN+WN", 1):
            x, n0 = G[k]
            T = sinefit(x[:n0])
            a0, s0 = sinamp(x[:n0], T); a1, s1 = sinamp(x[n0:], T)
            if a0 > .5 * s0:
                out.append((np.log(a1 / a0), np.log(s1 / s0)))
        o = np.array(out); big = np.abs(o[:, 1]) > .3
        slope = np.polyfit(o[big, 1], o[big, 0], 1)[0]
        print(f"S2 SIN breaks with |log noise ratio|>.3 (n={big.sum()}): slope of log amp ratio on log noise ratio {slope:+.2f} "
              f"(whole output scaled -> ~1, noise only -> ~0); corr {np.corrcoef(o[big, 1], o[big, 0])[0, 1]:+.2f}")
    if E == "S3":
        out = []
        for k in ids(("WN", "AR1"), 1):
            x, n0 = G[k]
            p0, p1 = ac(x[:n0]), ac(x[n0:])
            if abs(p1 - p0) > .2 and abs(x[n0:].mean() - x[:n0].mean()) < .2 * x[:n0].std():
                out.append((np.log(x[n0:].var() / x[:n0].var()), np.log((1 - p0 ** 2) / (1 - p1 ** 2))))
        o = np.array(out)
        print(f"S3 dep breaks |dphi|>.2, no level change (n={len(o)}): slope of log var ratio on log AR-var ratio {np.polyfit(o[:, 1], o[:, 0], 1)[0]:+.2f} "
              f"(not re-normalised -> ~1, re-normalised -> ~0); corr {np.corrcoef(o[:, 1], o[:, 0])[0, 1]:+.2f}")
    if E == "S4":
        rows = []
        for k in ids("RW"):
            x, n0 = G[k]
            d = np.diff(x); j = n0 - 1
            rest = np.delete(d, j)
            z = abs(d[j] - rest.mean()) / rest.std()
            db, da = d[:j], d[j + 1:]
            tt = abs(da.mean() - db.mean()) / np.sqrt(db.var() / len(db) + da.var() / len(da))
            sr = abs(np.log(da.std() / db.std()))
            rows.append((L[k], z, tt, sr))
        R = np.array(rows)
        print(f"S4 RW (n={len(R)}): label AUC  increment z at n0 {roc_auc_score(R[:, 0], R[:, 1]):.3f} | drift t {roc_auc_score(R[:, 0], R[:, 2]):.3f} | "
              f"step-sd log ratio {roc_auc_score(R[:, 0], R[:, 3]):.3f}  (output step -> z; drift on steps -> t; spread on steps -> sd ratio)")
    if E == "S5":
        rows = []
        for k in ids(("WN", "AR1", "AR2")):
            x, n0 = G[k]
            t = np.arange(len(x))
            r2 = np.corrcoef(t, x)[0, 1] ** 2
            rows.append((L[k], r2))
        R = np.array(rows)
        for y in (0, 1):
            v = R[R[:, 0] == y, 1]
            print(f"S5 WN/AR label {y}: share with linear-trend R2 > .2: {np.mean(v > .2):.3f}, > .4: {np.mean(v > .4):.3f}")
    if E == "S6":
        rows = []
        for k in ids("WN"):
            x, n0 = G[k]
            a, b = x[:n0], x[n0:]
            if abs(b.mean() - a.mean()) < .1 * a.std() and abs(np.log(b.std() / a.std())) < .05 and abs(ac(b) - ac(a)) < .05:
                za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
                rows.append((L[k], abs(kurtosis(zb) - kurtosis(za)), ks_2samp(za, zb).statistic))
        R = np.array(rows)
        print(f"S6 WN with unchanged mean/sd/acf (n={len(R)}, label-1 share {R[:, 0].mean():.3f}): label AUC |d kurt| {roc_auc_score(R[:, 0], R[:, 1]):.3f}, "
              f"standardised KS {roc_auc_score(R[:, 0], R[:, 2]):.3f}  (.5 = no shape mechanism visible)")
    if E == "S7":
        rows = []
        for k in ids("WN", 0):
            x, n0 = G[k]
            dk = abs(kurtosis(x[n0:]) - kurtosis(x[:n0]))
            h = n0 // 2
            pk = abs(kurtosis(x[h:n0]) - kurtosis(x[:h])) if n0 > 200 else np.nan
            rows.append((dk, pk, min(n0, len(x) - n0)))
        R = pd.DataFrame(rows, columns=["at", "pseudo", "m"]).dropna()
        print(f"S7 WN label 0 (n={len(R)}): median |d kurt| at boundary {R['at'].median():.3f} vs pseudo split in pre {R.pseudo.median():.3f} "
              f"(equal -> no noise re-draw at n0 in no-break series)")


for E in sys.argv[1:]:
    run(E)
