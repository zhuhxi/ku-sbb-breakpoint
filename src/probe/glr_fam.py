"""Generalised (profile) likelihood ratios per generator family x break kind, model-averaged with the semi-automatic family
probabilities (expert recipe: profile LR at the known date + Bayesian model averaging over model families).
Each family is a linear regression y = Z b + s e, e ~ unit-variance t(nu) (nu profiled on a grid incl. Gaussian):
  WN   y = x                 Z = [1]
  AR   y = x                 Z = [1, x(t-1), x(t-2)]
  SIN  y = x                 Z = [1, sin, cos at the pre-segment FFT period, x(t-1)]
  RW   y = diff(x)           Z = [1]                       (level break = one big increment at n0)
  LL   y = diff(x)           Z = [1, d(t-1), d(t-2), d(t-3)]
H0: one model for the whole series. H1 per kind: level (step regressor; impulse for RW/LL), spread (own scale per segment),
dep (lag / harmonic coefficients interacted with the post indicator), shape (own nu per segment). glr = 2 (l1 - l0).
Averaged columns: sum_f p(f | x) glr_{f,kind} with p from outputs/factory/semiauto_bag6.parquet (saf_family_*; AR = AR1 + AR2).
    python probe/glr_fam.py  -> outputs/factory/glrfam.parquet"""
import time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.special import gammaln
NUS = (3., 4., 6., 10., 30., np.inf)


def tll(r, v=None):
    """max over nu of the t log-likelihood of residuals r with moment-matched scale (variance v fixed if given)."""
    n = len(r); v = (r.var() + 1e-12) if v is None else v; best = -np.inf
    for nu in NUS:
        if np.isinf(nu):
            ll = -.5 * n * (np.log(2 * np.pi * v) + 1)
        else:
            s2 = v * (nu - 2) / nu
            ll = n * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - .5 * np.log(nu * np.pi * s2)) - (nu + 1) / 2 * np.log1p(r * r / (nu * s2)).sum()
        best = max(best, ll)
    return best


def ols(y, Z):
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def design(x, n0, fam):
    if fam in ("RW", "LL"):
        y = np.diff(x); b = n0 - 1
    else:
        y = x.astype(float); b = n0
    lag = {"WN": 0, "AR": 2, "SIN": 1, "RW": 0, "LL": 3}[fam]
    cols = [np.ones(len(y))]
    if fam == "SIN":
        z = x[:n0] - x[:n0].mean(); P = np.abs(np.fft.rfft(z)) ** 2; k = P[1:].argmax() + 1; T = n0 / k
        t = np.arange(len(y)); cols += [np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T)]
    for j in range(1, lag + 1):
        cols.append(np.r_[np.full(j, y[0]), y[:-j]])
    Z = np.column_stack(cols)[lag:]; y = y[lag:]; b -= lag
    return y, Z, max(b, 5)


def fam_glr(x, n0, fam):
    y, Z, b = design(x, n0, fam); post = (np.arange(len(y)) >= b).astype(float)
    r0 = ols(y, Z); l0 = tll(r0)
    step = np.zeros(len(y)); step[b] = 1.0
    lvl = ols(y, np.column_stack([Z, step if fam in ("RW", "LL") else post]))
    out = {f"glr_{fam}_level": 2 * (tll(lvl) - l0)}
    out[f"glr_{fam}_spread"] = 2 * (tll(r0[:b]) + tll(r0[b:]) - l0)   # spread: own scale per segment
    Zd = Z[:, 1:] if Z.shape[1] > 1 else np.r_[y[:1], y[:-1]][:, None]
    dep = ols(y, np.column_stack([Z if Z.shape[1] > 1 else np.column_stack([Z, Zd]), Zd * post[:, None]]))
    out[f"glr_{fam}_dep"] = 2 * (tll(dep) - l0)
    v = r0.var() + 1e-12   # shape: common scale, own nu per segment
    out[f"glr_{fam}_shape"] = 2 * (tll(r0[:b], v) + tll(r0[b:], v) - tll(r0, v))
    return out


def one(x, n0):
    o = {}
    for f in ("WN", "AR", "SIN", "RW", "LL"):
        try: o.update(fam_glr(x, n0, f))
        except Exception: pass
    return o


if __name__ == "__main__":
    import sys; sys.path.insert(0, "probe"); import glr_fam as G
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"])
    items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    F = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(G.one)(x, n0) for _, x, n0 in items), index=[k for k, _, _ in items]).astype(float)
    A = pd.read_parquet("outputs/factory/semiauto_bag6.parquet").reindex(F.index)
    pf = {"WN": A.saf_family_WN, "AR": A.saf_family_AR1 + A.saf_family_AR2, "SIN": A.saf_family_SIN, "RW": A.saf_family_RW, "LL": A.saf_family_LL}
    tot = sum(pf.values())
    for k in ("level", "spread", "dep", "shape"):
        F[f"glrbma_{k}"] = sum(pf[f] / tot * F[f"glr_{f}_{k}"] for f in pf)
    F["glrbma_max"] = F[[f"glrbma_{k}" for k in ("level", "spread", "dep", "shape")]].max(1)
    F.to_parquet("outputs/factory/glrfam.parquet"); print(F.shape, f"{time.time() - t0:.0f}s")
    L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    from sklearn.metrics import roc_auc_score
    print("alone AUC (top 8):", sorted(((round(roc_auc_score(L, F.loc[L.index, c].fillna(0)), 3), c) for c in F), reverse=True)[:8])
