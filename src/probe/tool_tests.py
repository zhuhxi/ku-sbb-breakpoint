"""Tool fingerprints of the official code (real no-break training series, structure from genA).
T1 t noise standardised to unit variance (standard_t / sqrt(df/(df-2))) or raw standard_t?  heavy-tailed WN series
   would be ~1.3x wider if raw -> compare log2 sd of high- vs low-kurtosis WN series (the shell scale is the same).
T2 sine phase: uniform over the circle (random phase) or fixed?
T3 start of AR series: burn-in (stationary start) or zero start (statsmodels arma_generate_sample default burnin=0)?
T4 sine periods: range of the fitted integer periods.
T5 shell order: base + size * core (|mean| independent of sd) or size * (core + base) (|mean| grows with sd)?
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, kstest, ks_2samp

STRUCT = ["WN", "AR1", "AR2", "MA1", "ARMA11", "RW", "SIN+WN", "SIN+AR1"]
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
A = pd.read_parquet("outputs/factory/genA.parquet"); A.index = A.index.astype(str)
st = A["g_struct"].reindex(L.index).map(dict(enumerate(STRUCT)))
S = pd.read_parquet("data/train_series.parquet"); S["id"] = S["id"].astype(str)
neg = set(L.index[L == 0])
S = S[S.id.isin(neg)].sort_values(["id", "time"])
G = {k: g["value"].to_numpy(np.float64) for k, g in S.groupby("id", sort=False)}
wn = [k for k in G if st[k] == "WN"]
ku = np.array([kurtosis(G[k]) for k in wn]); sd = np.array([np.log2(G[k].std()) for k in wn]); mu = np.array([G[k].mean() for k in wn])
hi, lo = sd[ku > 1.0], sd[np.abs(ku) < .15]
print(f"T1 WN log2 sd: heavy-tailed (kurt>1, n={len(hi)}) median {np.median(hi):+.3f} vs Gaussian-like (n={len(lo)}) {np.median(lo):+.3f}; KS p {ks_2samp(hi, lo).pvalue:.3f}  (raw t would shift heavy ones up by ~+.3)")
ph, per = [], []
for k in [k for k in G if st[k] == "SIN+WN"]:
    x = G[k]; t = np.arange(len(x)); z = x - x.mean()
    P = np.abs(np.fft.rfft(z)) ** 2; kk = P[1:].argmax() + 1
    best = None
    for T in np.arange(11.5, 33.5, .05):
        M = np.column_stack([np.ones(len(x)), np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T)])
        c, res, *_ = np.linalg.lstsq(M, x, rcond=None)
        rr = ((x - M @ c) ** 2).sum()
        if best is None or rr < best[0]:
            best = (rr, T, np.arctan2(c[2], c[1]))
    per.append(best[1]); ph.append(best[2])
ph, per = np.array(ph), np.array(per)
print(f"T2 SIN phase (n={len(ph)}): KS vs uniform(-pi, pi) p {kstest((ph + np.pi) / (2 * np.pi), 'uniform').pvalue:.3f}")
near = np.abs(per - np.round(per))
print(f"T4 SIN fitted period min {per.min():.2f} max {per.max():.2f}; share within .1 of an integer {np.mean(near < .1):.3f} (uniform would be .2); integer counts 11-13: {[(int(v), int(np.sum(np.round(per) == v))) for v in (11, 12, 13)]}, 32-34: {[(int(v), int(np.sum(np.round(per) == v))) for v in (32, 33, 34)]}")
ratios = []
for k in [k for k in G if st[k] == "AR1"]:
    x = G[k]; z = x - x.mean(); ph1 = z[1:] @ z[:-1] / (z[:-1] @ z[:-1])
    if ph1 > .7 and len(x) > 300:
        ratios.append(np.log(x[:30].var() / x[100:].var()))
r = np.array(ratios)
print(f"T3 AR1 phi>.7 (n={len(r)}): median log var(first 30)/var(rest) {np.median(r):+.3f}  (zero start -> clearly negative; burn-in -> ~0 or slightly negative from short-window bias)")
c = np.corrcoef(np.abs(mu), 2 ** sd)[0, 1]
print(f"T5 WN corr(|mean|, sd) {c:+.3f}  (base + size*core -> ~0; size*(core + base) -> clearly positive)")
rs, rc = [], []
for k in [k for k in G if st[k] == "AR1"]:
    x = G[k]; z = x - x.mean(); ph1 = z[1:] @ z[:-1] / (z[:-1] @ z[:-1])
    if ph1 > .7 and len(x) > 400:
        rest = x[100:]
        rs.append(np.log(x[:30].var() / rest.var())); rc.append(np.log(x[200:230].var() / rest.var()))
print(f"T3b same 30-point window at the start vs at t=200 (control, n={len(rs)}): median {np.median(rs):+.3f} vs {np.median(rc):+.3f}  (equal -> burn-in; start clearly lower -> zero start)")
