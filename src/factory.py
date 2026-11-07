"""Feature factory in the style of datatech (ADIA 2025, LB 0.886): statistic x transform x template.

    python factory.py st cost          # build families, one parquet each: outputs/factory/{fam}.parquet
    python factory.py st ABSD RES SQ    # same family on the extra transforms -> outputs/factory/st_ABSD_RES_SQ.parquet

A family is a function f(x, n0) registered with @register. It returns a dict whose values are either
  * a (left, right, whole) triple  -> expanded by the 9-column template
        _l _r _w _diff (r-l) _ratio (r/l) _cl (l/(l+r)) _cr (r/(l+r)) _rwl (l/w) _rwr (r/w)
  * a scalar                       -> kept as is.
Every family runs on three transforms of each series: RAW, DIFF (prepend first value, same length, boundary
stays at n0) and CUMSUM. Columns are named {fam}_{TRANSFORM}_{stat}{suffix}; rows are train + test ids.
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as ss
from scipy.ndimage import maximum_filter1d

REG = {}
OUT = Path("outputs/factory")


def register(f):
    REG[f.__name__] = f
    return f


def seg3(func, x, n0):
    return (func(x[:n0]), func(x[n0:]), func(x))


# ---------------------------------------------------------------- batch 1
def _rstd50(s):
    if len(s) < 50:
        return np.nan
    c1, c2 = np.cumsum(np.r_[0, s]), np.cumsum(np.r_[0, s * s])
    m = (c1[50:] - c1[:-50]) / 50
    return np.sqrt(np.maximum((c2[50:] - c2[:-50]) / 50 - m * m, 0)).mean()


def _slope(s):
    t = np.arange(len(s)) - (len(s) - 1) / 2
    return (t @ (s - s.mean())) / (t @ t)


@register
def st(x, n0):
    fs = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min, "range": np.ptp, "std": np.std,
          "skew": ss.skew, "kurt": ss.kurtosis, "cv": lambda s: s.std() / s.mean() if abs(s.mean()) > 1e-6 else 0.0,
          "rstd50": _rstd50, "slope": _slope, "q10": lambda s: np.quantile(s, 0.1), "q90": lambda s: np.quantile(s, 0.9)}
    return {k: seg3(f, x, n0) for k, f in fs.items()}


def _ar_sse(s, p=4):
    if len(s) <= 2 * p + 2:
        return np.nan
    X = np.column_stack([np.ones(len(s) - p)] + [s[p - i:-i] for i in range(1, p + 1)])
    b, *_ = np.linalg.lstsq(X, s[p:], rcond=None)
    e = s[p:] - X @ b
    return e @ e


def _lin_sse(s):
    t = np.arange(len(s), dtype=float)
    b = np.polyfit(t, s, 1)
    e = s - np.polyval(b, t)
    return e @ e


@register
def cost(x, n0):
    """ruptures-style segment costs (lower = better fit), plus the split gain (w - l - r) / w."""
    sd = x.std() + 1e-12
    z = (x - x.mean()) / sd
    rng = np.random.default_rng(0)
    W, B = rng.normal(0, 1.0, 64), rng.uniform(0, 2 * np.pi, 64)       # RBF kernel cost via random Fourier features
    phi = np.sqrt(2 / 64) * np.cos(np.outer(z, W) + B)
    r = ss.rankdata(x)
    r = (r - r.mean()) / (r.std() + 1e-12)
    f = {
        "l1": lambda a, b: np.abs(a[b] - np.median(a[b])).sum(),
        "l2": lambda a, b: ((a[b] - a[b].mean()) ** 2).sum(),
        "normal": lambda a, b: (b.stop - b.start if isinstance(b, slice) else len(a)) * np.log(a[b].var() + 1e-12),
        "ar4": lambda a, b: _ar_sse(a[b]),
        "lin": lambda a, b: _lin_sse(a[b]),
    }
    out = {}
    L, R, Wh = slice(0, n0), slice(n0, len(x)), slice(0, len(x))
    for k, fn in f.items():
        trip = (fn(z, L), fn(z, R), fn(z, Wh))
        out[k] = trip
        out[k + "_gain"] = (trip[2] - trip[0] - trip[1]) / (abs(trip[2]) + 1e-12)
    rb = lambda b: len(phi[b]) - (phi[b].sum(0) ** 2).sum() / len(phi[b])
    out["rbf"] = (rb(L), rb(R), rb(Wh))
    out["rbf_gain"] = (out["rbf"][2] - out["rbf"][0] - out["rbf"][1]) / (out["rbf"][2] + 1e-12)
    rk = lambda b: -(r[b].sum() ** 2) / len(r[b])
    out["rank"] = (rk(L), rk(R), rk(Wh))
    return out


# ---------------------------------------------------------------- batch 2
def _pacf(s, lags):
    s = s - s.mean()
    n = len(s)
    ac = np.array([1.0] + [(s[:-k] @ s[k:]) / (s @ s + 1e-12) for k in range(1, max(lags) + 1)])
    out, phi = {}, np.zeros(0)
    for k in range(1, max(lags) + 1):        # Durbin-Levinson
        a = (ac[k] - phi @ ac[1:k][::-1]) / (1 - phi @ ac[1:k] + 1e-12)
        phi = np.r_[phi - a * phi[::-1], a]
        if k in lags:
            out[k] = a
    return out


def _chunks(s, m, agg):
    k = len(s) // m
    if k < 3:
        return np.nan, np.nan, np.nan
    v = getattr(s[:k * m].reshape(k, m), agg)(1)
    lr = ss.linregress(np.arange(k), v)
    return lr.slope, lr.rvalue, lr.intercept


def _change_q(s, ql, qh, isabs, agg):
    lo, hi = np.quantile(s, [ql, qh])
    inside = (s >= lo) & (s <= hi)
    both = inside[1:] & inside[:-1]
    d = np.diff(s)[both]
    if isabs:
        d = np.abs(d)
    return getattr(d, agg)() if len(d) > 1 else 0.0


def _benford(s):
    a = np.abs(s[s != 0])
    fd = (a / 10 ** np.floor(np.log10(a))).astype(int)
    obs = np.bincount(np.clip(fd, 1, 9), minlength=10)[1:] / max(len(fd), 1)
    ben = np.log10(1 + 1 / np.arange(1, 10))
    return np.corrcoef(obs, ben)[0, 1] if obs.std() > 0 else 0.0


def _peaks(s, n):
    m = maximum_filter1d(s, 2 * n + 1, mode="constant", cval=-np.inf)
    return int(((s == m)[n:-n]).sum()) if len(s) > 2 * n else 0


@register
def tsf(x, n0):
    out = {}
    g = lambda f: seg3(f, x, n0)
    for r in (0.5, 1, 1.5, 3):
        out[f"beyond{r}"] = g(lambda s: np.mean(np.abs(s - s.mean()) > r * s.std()))
    out["above0"] = g(lambda s: np.mean(s > 0))
    for n in (5, 25):
        out[f"peaks{n}"] = g(lambda s: _peaks(s, n) / len(s))
    for k in (2, 4, 6):
        out[f"pacf{k}"] = g(lambda s: _pacf(s, (k,))[k])
    for q in (0.1, 0.6, 0.8):
        out[f"imq{q}"] = g(lambda s: np.argmax(np.cumsum(np.abs(s)) >= q * np.abs(s).sum()) / len(s))
    out["erc10_9"] = g(lambda s: (np.array_split(s, 10)[9] ** 2).sum() / (s @ s + 1e-12))
    out["erc20_16"] = g(lambda s: (np.array_split(s, 20)[16] ** 2).sum() / (s @ s + 1e-12))
    lt = {"slope": 0, "rvalue": 2, "pvalue": 3, "intercept": 1}
    for a, i in lt.items():
        out[f"lt_{a}"] = g(lambda s: ss.linregress(np.arange(len(s)), s)[i])
    for m in (5, 10, 50):
        for agg in ("mean", "max"):
            for j, a in enumerate(("slope", "rvalue", "intercept")):
                out[f"alt{m}{agg}_{a}"] = g(lambda s: _chunks(s, m, agg)[j])
    for c in (1, 2, 3):
        out[f"fftim{c}"] = g(lambda s: np.fft.rfft(s)[c].imag / len(s))
    for ql, qh, isabs, agg in [(0.4, 1.0, True, "var"), (0.2, 1.0, True, "var"), (0.6, 0.8, True, "var"), (0.2, 0.4, True, "var"),
                               (0.4, 1.0, False, "var"), (0.0, 0.8, False, "var"), (0.4, 0.6, False, "var"), (0.2, 0.8, False, "var"),
                               (0.4, 1.0, True, "mean"), (0.4, 0.6, True, "mean"), (0.0, 0.2, False, "mean"), (0.8, 1.0, False, "mean")]:
        out[f"cq{ql}_{qh}_{int(isabs)}{agg}"] = g(lambda s: _change_q(s, ql, qh, isabs, agg))
    out["argmax"] = g(lambda s: np.argmax(s) / len(s))
    out["argmin"] = g(lambda s: np.argmin(s) / len(s))
    out["benford"] = g(_benford)
    return out


def _sampen(s, m=2, rr=0.2):
    s = s[:600]
    r = rr * s.std()
    if r == 0 or len(s) < 20:
        return np.nan
    def cnt(mm):
        E = np.lib.stride_tricks.sliding_window_view(s, mm)[: len(s) - m]
        D = np.abs(E[:, None, :] - E[None, :, :]).max(2)
        return ((D <= r).sum() - len(E)) / 2
    a, b = cnt(m + 1), cnt(m)
    return -np.log((a + 1) / (b + 1))


def _hjorth(s):
    d1, d2 = np.diff(s), np.diff(s, 2)
    v0, v1, v2 = s.var() + 1e-12, d1.var() + 1e-12, d2.var() + 1e-12
    mob = np.sqrt(v1 / v0)
    return mob, np.sqrt(v2 / v1) / mob


def _higuchi(s, kmax=10):
    N = len(s)
    lk = []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            Lm.append(np.abs(np.diff(s[idx])).sum() * (N - 1) / ((len(idx) - 1) * k) / k)
        lk.append(np.mean(Lm))
    return -np.polyfit(np.log(np.arange(1, kmax + 1)), np.log(np.array(lk) + 1e-12), 1)[0]


def _petrosian(s):
    d = np.diff(s)
    nd = (d[1:] * d[:-1] < 0).sum()
    n = len(s)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * nd)))


def _katz(s):
    d = np.abs(np.diff(s))
    L, a = d.sum(), d.mean() + 1e-12
    dist = np.abs(s - s[0]).max() + 1e-12
    return np.log10(L / a) / np.log10(dist / a) if dist > a else np.nan


def _svdent(s, order=3, delay=1):
    E = np.lib.stride_tricks.sliding_window_view(s, order)
    w = np.linalg.svd(E - E.mean(0), compute_uv=False)
    w = w / (w.sum() + 1e-12)
    return -(w * np.log2(w + 1e-12)).sum()


def _specent(s):
    p = np.abs(np.fft.rfft(s - s.mean()))[1:] ** 2
    p = p / (p.sum() + 1e-12)
    return -(p * np.log2(p + 1e-12)).sum() / np.log2(len(p))


def _permen(s, order):
    E = np.lib.stride_tricks.sliding_window_view(s, order)
    code = (np.argsort(E, 1) * order ** np.arange(order)).sum(1)
    _, c = np.unique(code, return_counts=True)
    p = c / c.sum()
    return -(p * np.log(p)).sum() / np.log(math.factorial(order))


def _condent(s, lag=1):
    b = (s > np.median(s)).astype(int)
    a, c = b[:-lag], b[lag:]
    h = 0.0
    for v in (0, 1):
        m = a == v
        if m.sum() == 0:
            continue
        p1 = c[m].mean()
        hh = -sum(q * np.log2(q) for q in (p1, 1 - p1) if q > 0)
        h += m.mean() * hh
    return h


@register
def ent(x, n0):
    g = lambda f: seg3(f, x, n0)
    hj = seg3(_hjorth, x, n0)
    return {"sampen": g(_sampen), "hj_mob": tuple(h[0] for h in hj), "hj_cplx": tuple(h[1] for h in hj),
            "higuchi": g(_higuchi), "petro": g(_petrosian), "katz": g(_katz), "svdent": g(_svdent), "specent": g(_specent),
            "pe3": g(lambda s: _permen(s, 3)), "pe4": g(lambda s: _permen(s, 4)), "condent": g(_condent)}


# ---------------------------------------------------------------- batch 3
def _arfit(s, p=5):
    X = np.column_stack([np.ones(len(s) - p)] + [s[p - i:-i] for i in range(1, p + 1)])
    b, *_ = np.linalg.lstsq(X, s[p:], rcond=None)
    e = s[p:] - X @ b
    n, k = len(e), p + 1
    sig2 = e @ e / n + 1e-300
    return b, e, n * np.log(sig2) + 2 * k, n * np.log(sig2) + k * np.log(n)


def _onestep(b, s, p=5):
    X = np.column_stack([np.ones(len(s) - p)] + [s[p - i:-i] for i in range(1, p + 1)])
    return s[p:] - X @ b


def _dynamic(b, hist, n, p=5):
    h = list(hist[-p:])
    out = np.empty(n)
    for t in range(n):
        v = b[0] + b[1:] @ np.array(h[::-1][:p])
        out[t] = v
        h.append(v)
    return out


@register
def ar2(x, n0):
    a, b = x[:n0], x[n0:]
    out = {}
    fa, fb, fw = _arfit(a), _arfit(b), _arfit(x)
    for nm, i in (("aic", 2), ("bic", 3)):
        out[nm] = (fa[i] / len(a), fb[i] / len(b), fw[i] / len(x))
    out["rstd"] = (fa[1].std(), fb[1].std(), fw[1].std())
    for j in range(6):
        out[f"p{j}"] = (fa[0][j], fb[0][j], fw[0][j])
    for tag, f_src, src, tgt, s_src in (("ab", fa, a, b, fa[1].std()), ("ba", fb, b, a, fb[1].std())):
        tgt_o = tgt if tag == "ab" else tgt[::-1]
        src_o = src if tag == "ab" else src[::-1]
        bb = f_src[0] if tag == "ab" else _arfit(src_o)[0]
        sd = s_src + 1e-12
        e1 = _onestep(bb, np.r_[src_o[-5:], tgt_o]) / sd
        out[f"{tag}_os_mean"], out[f"{tag}_os_std"] = e1.mean(), e1.std()
        out[f"{tag}_os_skew"], out[f"{tag}_os_kurt"] = ss.skew(e1), ss.kurtosis(e1)
        out[f"{tag}_os_absmean"] = np.abs(e1).mean()
        dyn = (tgt_o[:50] - _dynamic(bb, src_o, min(50, len(tgt_o)))) / (src_o.std() + 1e-12)
        out[f"{tag}_dyn_mean"], out[f"{tag}_dyn_std"] = dyn.mean(), dyn.std()
    return out


@register
def normt(x, n0):
    sh = lambda s: ss.shapiro(s[:2000])[1]
    jb = lambda s: ss.jarque_bera(s)[1]
    da = lambda s: ss.normaltest(s)[0]
    return {"shapiro_p": seg3(sh, x, n0), "jb_p": seg3(jb, x, n0), "dagost": seg3(da, x, n0)}


# ---------------------------------------------------------------- batch 4
@register
def near(x, n0):
    """Template stats on windows at the boundary: left = last w before, right = first w after, whole = both."""
    out = {}
    fs = {"mean": np.mean, "std": np.std, "median": np.median, "mad": lambda s: np.median(np.abs(s - np.median(s))),
          "absd": lambda s: np.abs(np.diff(s)).mean(), "acf1": lambda s: np.corrcoef(s[1:], s[:-1])[0, 1],
          "q10": lambda s: np.quantile(s, 0.1), "q90": lambda s: np.quantile(s, 0.9), "slope": _slope}
    for w in (15, 50, 150):
        a, b = x[max(n0 - w, 0):n0], x[n0:n0 + w]
        for k, f in fs.items():
            out[f"w{w}_{k}"] = (f(a), f(b), f(np.r_[a, b]))
    return out


@register
def acfm(x, n0):
    """Autocorrelation profile (lags 1-10, 20) and variance ratios per segment."""
    def acf(s, k):
        s = s - s.mean()
        return (s[:-k] @ s[k:]) / (s @ s + 1e-12)
    out = {f"acf{k}": seg3(lambda s: acf(s, k), x, n0) for k in (1, 2, 3, 4, 5, 7, 10, 20)}
    for q in (2, 4, 8, 16):
        out[f"vr{q}"] = seg3(lambda s: np.var(s[q:] - s[:-q]) / (q * np.var(np.diff(s)) + 1e-12), x, n0)
    out["acf_sq1"] = seg3(lambda s: acf((s - s.mean()) ** 2, 1), x, n0)
    out["acf_abs1"] = seg3(lambda s: acf(np.abs(s - s.mean()), 1), x, n0)
    return out


@register
def fe(x, n0, orders=(1, 5)):
    """Out-of-sample prediction error, both directions (ab: fit before, predict after; ba: fit the reversed after,
    predict the reversed before). Compares the out-of-sample one-step residuals o with the in-sample residuals e."""
    out = {}
    for tag, src, tgt in (("ab", x[:n0], x[n0:]), ("ba", x[n0:][::-1], x[:n0][::-1])):
        for p in orders:
            b, e, _, _ = _arfit(src, p)
            o = _onestep(b, np.r_[src[-p:], tgt], p)
            sd = e.std() + 1e-12
            z = o / sd
            k = f"{tag}{p}"
            out[f"{k}_lvar"] = np.log(o.var() / sd ** 2 + 1e-12)
            out[f"{k}_lmad"] = np.log(np.median(np.abs(o - np.median(o))) / (np.median(np.abs(e - np.median(e))) + 1e-12) + 1e-12)
            out[f"{k}_t"] = z.mean() * np.sqrt(len(z))
            out[f"{k}_ks"] = ss.ks_2samp(e / sd, z).statistic
            c = np.cumsum(z - 0) / np.sqrt(len(z))
            out[f"{k}_cusum"] = np.abs(c).max()
            c2 = np.cumsum(z ** 2 - 1) / np.sqrt(2 * len(z))
            out[f"{k}_cusum2"] = np.abs(c2).max()
            zz = z - z.mean()
            out[f"{k}_acf1"] = (zz[1:] @ zz[:-1]) / (zz @ zz + 1e-12)
            q = z ** 2 - (z ** 2).mean()
            out[f"{k}_sqacf1"] = (q[1:] @ q[:-1]) / (q @ q + 1e-12)
            out[f"{k}_kurt"] = ss.kurtosis(z) - ss.kurtosis(e / sd)
            n20 = min(20, len(z))
            out[f"{k}_near20"] = np.mean(z[:n20] ** 2)
    hi = max(orders)
    out["sym_lvar"] = out[f"ab{hi}_lvar"] - out[f"ba{hi}_lvar"]
    out["max_lvar"] = max(abs(out[f"ab{hi}_lvar"]), abs(out[f"ba{hi}_lvar"]))
    return out


@register
def fe2(x, n0):
    """fe with higher AR orders (10, 20)."""
    return fe(x, n0, orders=(10, 20))


@register
def fe3(x, n0):
    """fe with AR orders (40, 60); needs both segments longer than ~2 * order (min segment here is ~100)."""
    return fe(x, n0, orders=(40,))


@register
def rst(x, n0):
    """Tail / scale statistics meant for residual transforms: fine quantiles, robust scales, exceedance rates
    against the whole-series scale, tail means, and the position of the largest values."""
    sw = np.std(x) + 1e-12
    mw = np.median(x)
    fs = {f"q{int(q * 100):02d}": (lambda q: lambda s: np.quantile(s, q))(q) for q in (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)}
    fs.update({
        "iqr": lambda s: np.subtract(*np.quantile(s, [0.75, 0.25])),
        "mad": lambda s: np.median(np.abs(s - np.median(s))),
        "tmean": lambda s: ss.trim_mean(s, 0.1),
        "ex2": lambda s: np.mean(np.abs(s - mw) > 2 * sw), "ex3": lambda s: np.mean(np.abs(s - mw) > 3 * sw),
        "top5": lambda s: np.sort(s)[-max(len(s) // 20, 1):].mean(),
        "bot5": lambda s: np.sort(s)[: max(len(s) // 20, 1)].mean(),
        "ms": lambda s: np.mean(s ** 2), "l1": lambda s: np.mean(np.abs(s)),
        "head10": lambda s: np.mean(s[: max(len(s) // 10, 2)] ** 2) / (np.mean(s ** 2) + 1e-12),
        "tail10": lambda s: np.mean(s[-max(len(s) // 10, 2):] ** 2) / (np.mean(s ** 2) + 1e-12),
        "argmaxabs": lambda s: np.argmax(np.abs(s - np.median(s))) / len(s),
    })
    return {k: seg3(f, x, n0) for k, f in fs.items()}


def _arch(s, q=2):
    """OLS of s_t^2 on [1, s_{t-1}^2 .. s_{t-q}^2]: intercept, coefficients, persistence, unconditional level."""
    e2 = (s - s.mean()) ** 2
    X = np.column_stack([np.ones(len(e2) - q)] + [e2[q - i:-i] for i in range(1, q + 1)])
    b, *_ = np.linalg.lstsq(X, e2[q:], rcond=None)
    pers = b[1:].sum()
    return b[0], b[1], b[2], pers, b[0] / max(1 - pers, 0.05)


@register
def arch(x, n0):
    """ARCH(2) fitted per segment and on the whole series (meant for residual transforms)."""
    r = seg3(_arch, x, n0)
    names = ("a0", "a1", "a2", "pers", "uncond")
    return {k: tuple(t[i] for t in r) for i, k in enumerate(names)}


def _pe_stats(src, tgt, p):
    b, e, _, _ = _arfit(src, p)
    o = _onestep(b, np.r_[src[-p:], tgt], p)
    sd = e.std() + 1e-12
    z = o / sd
    q = z ** 2 - (z ** 2).mean()
    return {"lvar": np.log(o.var() / sd ** 2 + 1e-12), "t": z.mean() * np.sqrt(len(z)),
            "ks": ss.ks_2samp(e / sd, z).statistic, "cusum2": np.abs(np.cumsum(z ** 2 - 1)).max() / np.sqrt(2 * len(z)),
            "sqacf1": (q[1:] @ q[:-1]) / (q @ q + 1e-12)}


@register
def fez(x, n0):
    """Prediction error at the real boundary vs placebo splits inside each segment (same fit/test proportions).
    Columns: real, placebo before (pl), placebo after (pr), and real minus the larger placebo (in absolute value)."""
    out = {}
    a, b = x[:n0], x[n0:]
    for p in (1, 5):
        real = _pe_stats(a, b, p)
        pl = _pe_stats(a[: len(a) // 2], a[len(a) // 2:], p)
        pr = _pe_stats(b[: len(b) // 2], b[len(b) // 2:], p)
        for k in real:
            out[f"p{p}_{k}"] = real[k]
            out[f"p{p}_{k}_pl"], out[f"p{p}_{k}_pr"] = pl[k], pr[k]
            out[f"p{p}_{k}_ex"] = abs(real[k]) - max(abs(pl[k]), abs(pr[k]))
    return out


# ---------------------------------------------------------------- batch 6 (after the v3 submission)
@register
def tt(x, n0):
    """Two-sample test statistics, before vs after (scalars): location, scale and distribution tests.
    Meant for residual transforms, where the segments are close to i.i.d. and the tests are well calibrated."""
    a, b = x[:n0], x[n0:]
    out = {}
    out["welch_t"] = ss.ttest_ind(a, b, equal_var=False).statistic
    u = ss.mannwhitneyu(a, b)
    out["mwu_z"] = (u.statistic - len(a) * len(b) / 2) / np.sqrt(len(a) * len(b) * (len(a) + len(b) + 1) / 12)
    out["lvar"] = np.log((b.var() + 1e-12) / (a.var() + 1e-12))
    out["levene"] = ss.levene(a, b).statistic
    out["levene_mean"] = ss.levene(a, b, center="mean").statistic
    out["ansari"] = ss.ansari(a, b).statistic / (len(a) * (len(a) + len(b) + 1) / 4)
    out["mood_z"] = ss.mood(a, b).statistic
    out["ks"] = ss.ks_2samp(a, b).statistic
    out["cvm"] = ss.cramervonmises_2samp(a, b).statistic
    out["ad"] = ss.anderson_ksamp([a, b]).statistic
    out["bm"] = ss.brunnermunzel(a, b).statistic
    ra, rb = np.abs(a - np.median(a)), np.abs(b - np.median(b))
    out["lmad"] = np.log((np.median(rb) + 1e-12) / (np.median(ra) + 1e-12))
    out["energy"] = ss.energy_distance(a, b) / (x.std() + 1e-12)
    out["wass"] = ss.wasserstein_distance(a, b) / (x.std() + 1e-12)
    return out


@register
def acfr(x, n0):
    """Residual-misfit profile: autocorrelations 1-5 and Ljung-Box Q(10) per segment (for whole-fit residual
    transforms, where a change in AR dynamics leaves opposite-signed autocorrelation in the two segments)."""
    def acf(s, k):
        s = s - s.mean()
        return (s[k:] @ s[:-k]) / (s @ s + 1e-12)
    def lb(s, m=10):
        n = len(s)
        return n * (n + 2) * sum(acf(s, k) ** 2 / (n - k) for k in range(1, m + 1))
    out = {f"acf{k}": seg3(lambda s, k=k: acf(s, k), x, n0) for k in range(1, 6)}
    out["lb10"] = seg3(lb, x, n0)
    out["sqlb10"] = seg3(lambda s: lb(s ** 2), x, n0)
    return out


@register
def spec(x, n0):
    """Spectral shape per segment: Hann-tapered periodogram pooled into 8 log-spaced bands (log power share),
    spectral centroid, flatness, log-log slope; plus scalar band-wise log ratios after/before and a spectral KL."""
    def bands(s):
        s = (s - s.mean()) * np.hanning(len(s))
        p = np.abs(np.fft.rfft(s))[1:] ** 2 + 1e-12
        f = np.arange(1, len(p) + 1) / (2 * len(p))
        edges = np.geomspace(f[0], 0.5 + 1e-9, 9)
        b = np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])]) + 1e-12
        return b / b.sum(), p, f
    bl, pl, fl = bands(x[:n0])
    br, pr, fr = bands(x[n0:])
    bw, pw, fw = bands(x)
    out = {}
    for i in range(8):
        out[f"b{i}"] = (np.log(bl[i]), np.log(br[i]), np.log(bw[i]))
        out[f"lr{i}"] = np.log(br[i] / bl[i])
    cen = lambda p, f: (p * f).sum() / p.sum()
    flat = lambda p: np.exp(np.log(p).mean()) / p.mean()
    slope = lambda p, f: np.polyfit(np.log(f), np.log(p), 1)[0]
    out["cen"] = (cen(pl, fl), cen(pr, fr), cen(pw, fw))
    out["flat"] = (flat(pl), flat(pr), flat(pw))
    out["slope"] = (slope(pl, fl), slope(pr, fr), slope(pw, fw))
    out["kl"] = float((br * np.log(br / bl)).sum() + (bl * np.log(bl / br)).sum())
    out["l1"] = float(np.abs(br - bl).sum())
    return out


@register
def fe4(x, n0):
    """Local prediction error: AR fitted on the last W points before the boundary, scored on the first H after
    (and the mirror), relative to the in-sample error; small windows isolate the boundary from slow drift."""
    out = {}
    for W, H, p in ((100, 30, 3), (150, 60, 5), (250, 100, 10)):
        for tag, src, tgt in (("ab", x[:n0], x[n0:]), ("ba", x[n0:][::-1], x[:n0][::-1])):
            s, t = src[-W:], tgt[:H]
            b, e, _, _ = _arfit(s, p)
            o = _onestep(b, np.r_[s[-p:], t], p)
            sd = e.std() + 1e-12
            k = f"{tag}W{W}"
            out[f"{k}_lvar"] = np.log(o.var() / sd ** 2 + 1e-12)
            out[f"{k}_t"] = (o / sd).mean() * np.sqrt(len(o))
            out[f"{k}_lmabs"] = np.log(np.abs(o).mean() / (np.abs(e).mean() + 1e-12) + 1e-12)
        out[f"W{W}_sym"] = out[f"abW{W}_lvar"] + out[f"baW{W}_lvar"]
    return out


@register
def pers(x, n0):
    """Persistence of the change (scalars): the after segment is cut into thirds a1..a3, the before segment into
    b1..b3. For each statistic, d_i = stat(a_i) - stat(before) in units of the before-thirds spread. Outputs the
    sign-consistent minimum (0 unless all three thirds moved the same way), the mean, the decay d1 - d3, the
    natural drift (max pairwise spread among the before thirds, and the after thirds), and persist / drift."""
    a, b = x[n0:], x[:n0]
    A, Bs = np.array_split(a, 3), np.array_split(b, 3)
    mad = lambda s: np.median(np.abs(s - np.median(s))) + 1e-12
    fs = {"med": np.median, "lsd": lambda s: np.log(s.std() + 1e-12), "lmad": lambda s: np.log(mad(s)),
          "acf1": lambda s: np.corrcoef(s[1:], s[:-1])[0, 1], "q90": lambda s: np.quantile(s, 0.9)}
    out = {}
    for k, f in fs.items():
        sb = f(b)
        tb = np.array([f(u) for u in Bs])
        ta = np.array([f(u) for u in A])
        drift_b = tb.max() - tb.min() + 1e-6
        d = (ta - sb) / drift_b
        same = np.all(d > 0) or np.all(d < 0)
        pm = np.sign(d[0]) * np.abs(d).min() if same else 0.0
        out[f"{k}_pmin"] = pm
        out[f"{k}_mean"] = d.mean()
        out[f"{k}_decay"] = d[0] - d[2]
        out[f"{k}_drift_a"] = (ta.max() - ta.min()) / drift_b
        out[f"{k}_raw_pmin"] = np.sign(ta[0] - sb) * np.abs(ta - sb).min() if same else 0.0
        out[f"{k}_jump"] = (ta[0] - tb[2]) / drift_b
    return out


def _rmed(s, w):
    return pd.Series(s).rolling(w, center=True, min_periods=1).median().to_numpy()


def _bimod(s):
    """Sarle's bimodality coefficient (0.555 for a uniform; higher = two clusters)."""
    return (ss.skew(s) ** 2 + 1) / (ss.kurtosis(s) + 3 + 1e-12)


def _gap2(m, s):
    """Two-means split of the smoothed level m: distance between the two centres in units of the fast noise."""
    c = np.sort(m)
    cand = range(len(c) // 10, len(c) - len(c) // 10, max(1, len(c) // 50))
    i = cand[int(np.argmax([i * (len(c) - i) * (c[:i].mean() - c[i:].mean()) ** 2 for i in cand]))]
    return abs(c[i:].mean() - c[:i].mean()) / (np.median(np.abs(s - m)) * 1.4826 + 1e-9)


@register
def regime(x, n0):
    """Slow level wandering / regime switching inside each segment (error analysis: missed breaks often sit on a
    wandering or switching background). Per segment, smoothed level m = rolling median (w = 15, 31, 61, computed
    within the segment): share of variance in m, crossings of its median per 100 points, bimodality, two-level
    gap, lag-w autocorrelation of m. Plus the level shift measured against the before-segment wandering."""
    out = {}
    b, a = x[:n0], x[n0:]
    for w in (15, 31, 61):
        def sv(s, w=w):
            m = _rmed(s, w)
            return m.var() / (s.var() + 1e-12)

        def cross(s, w=w):
            m = _rmed(s, w)
            c = np.sign(m - np.median(m))
            return 100 * np.count_nonzero(np.diff(c[c != 0])) / len(s)

        def bim(s, w=w):
            return _bimod(_rmed(s, w))

        def gap(s, w=w):
            return _gap2(_rmed(s, w), s)

        def lacf(s, w=w):
            m = _rmed(s, w)
            m = m - m.mean()
            return (m[w:] * m[:-w]).sum() / ((m * m).sum() + 1e-12) if len(m) > 2 * w else np.nan

        for k, f in {"sv": sv, "cross": cross, "bim": bim, "gap": gap, "lacf": lacf}.items():
            out[f"{k}{w}"] = seg3(f, x, n0)
        mb = _rmed(b, w)
        out[f"levw{w}"] = (np.median(a) - np.median(b)) / (mb.std() + 1e-6)
        out[f"levn{w}"] = (np.median(a) - np.median(b)) / (np.median(np.abs(b - mb)) * 1.4826 + 1e-9)
    return out


def _lagX(z, p):
    return np.column_stack([z[p - i:-i] for i in range(1, p + 1)]), z[p:]


@register
def dep(x, n0):
    """Dependence-only change (importance analysis: acf/AR features carry ~3% although 'how successive readings
    relate' is one of the three break types, and chow_f mixes in the level). Each segment is standardized by its
    own mean and sd first, so level and spread changes are removed. Scalars: AR(p) GLR pooled vs separate and
    a Wald test on the slope vectors (p = 1, 2, 3, 5, 10); PACF-difference z-scores (lags 1-5); variance-ratio
    log ratios (k = 2, 5, 10, 20); change of the lag-1..5 acf of |z| (volatility clustering)."""
    b, a = x[:n0], x[n0:]
    zb, za = (b - b.mean()) / (b.std() + 1e-12), (a - a.mean()) / (a.std() + 1e-12)
    out = {}
    for p in (1, 2, 3, 5, 10):
        Xb, yb = _lagX(zb, p)
        Xa, ya = _lagX(za, p)
        cb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        ca, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
        X, y = np.vstack([Xb, Xa]), np.r_[yb, ya]
        cp, *_ = np.linalg.lstsq(X, y, rcond=None)
        sb, sa = ((yb - Xb @ cb) ** 2).sum(), ((ya - Xa @ ca) ** 2).sum()
        sp = ((y - X @ cp) ** 2).sum()
        out[f"glr{p}"] = len(y) * np.log(sp / (sb + sa + 1e-12) + 1e-12)
        Vb = np.linalg.pinv(Xb.T @ Xb) * sb / max(len(yb) - p, 1)
        Va = np.linalg.pinv(Xa.T @ Xa) * sa / max(len(ya) - p, 1)
        d = ca - cb
        out[f"wald{p}"] = np.log1p(d @ np.linalg.pinv(Va + Vb) @ d)
        out[f"dsum{p}"] = ca.sum() - cb.sum()                    # persistence change (sum of AR slopes)
    se = np.sqrt(1 / len(b) + 1 / len(a))
    pa, pb = _pacf(za, range(1, 6)), _pacf(zb, range(1, 6))
    pz = np.array([pa[k] - pb[k] for k in range(1, 6)]) / se
    for j, v in enumerate(pz, 1):
        out[f"pz{j}"] = v
    out["pz_max"], out["pz_ss"] = np.abs(pz).max(), np.log1p((pz ** 2).sum())

    def vr(z, k):
        s = np.convolve(z, np.ones(k), "valid")
        return s.var() / (k * z.var() + 1e-12)

    for k in (2, 5, 10, 20):
        out[f"lvr{k}"] = np.log(vr(za, k) + 1e-6) - np.log(vr(zb, k) + 1e-6)
    ab, aa = np.abs(zb) - np.abs(zb).mean(), np.abs(za) - np.abs(za).mean()
    ac = lambda u, L: (u[L:] * u[:-L]).sum() / ((u * u).sum() + 1e-12)
    out["vc_d"] = sum(ac(aa, L) - ac(ab, L) for L in range(1, 6))
    out["vc_z"] = out["vc_d"] / (se * np.sqrt(5))
    return out


def _hampel(v, w=9, k=5.0):
    """Replace isolated spikes (|v - rolling median| > k robust sd) by the rolling median."""
    m = _rmed(v, w)
    r = v - m
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
    return np.where(np.abs(r) > k * s, m, v)


# ---------------------------------------------------------------- engine
def _ar_resid(v, p=5, fit_to=None, fit_from=None):
    """AR(p) one-step residuals of the whole series; coefficients fitted on v[:fit_to] (default: all of v),
    or on v[fit_from:] when fit_from is given."""
    X = np.column_stack([np.ones(len(v) - p)] + [v[p - i:-i] for i in range(1, p + 1)])
    m = len(X) if fit_to is None else fit_to - p
    sl = slice(fit_from - p, None) if fit_from is not None else slice(0, m)
    b, *_ = np.linalg.lstsq(X[sl], v[p:][sl], rcond=None)
    return np.r_[np.zeros(p), v[p:] - X @ b]


def _devol(v, p=10, lam=0.94):
    """AR(p) residual divided by the EWMA volatility of the PAST residuals (GARCH-like standardization)."""
    e = _ar_resid(v, p)
    s2 = np.empty(len(e))
    s2[0] = e[p:].var() + 1e-12
    for t in range(1, len(e)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * e[t - 1] ** 2
    return e / np.sqrt(s2 + 1e-12)


TRANSFORMS = {"RAW": lambda v, n0: v, "DIFF": lambda v, n0: np.diff(v, prepend=v[0]), "CUMSUM": lambda v, n0: np.cumsum(v)}
EXTRA_TRANSFORMS = {"ABSD": lambda v, n0: np.abs(np.diff(v, prepend=v[0])), "RES": lambda v, n0: _ar_resid(v),
                    "SQ": lambda v, n0: (v - np.median(v)) ** 2,
                    "RES1": lambda v, n0: _ar_resid(v, 1),                       # AR(1) residuals
                    "RESL": lambda v, n0: _ar_resid(v, 5, n0),                   # AR(5) fitted on the before segment only
                    "ARES": lambda v, n0: np.abs(_ar_resid(v)),                  # |residual|
                    "SRES": lambda v, n0: _ar_resid(v) ** 2,
                    "RES10": lambda v, n0: _ar_resid(v, 10), "RES20": lambda v, n0: _ar_resid(v, 20),
                    "ARES10": lambda v, n0: np.abs(_ar_resid(v, 10)), "ARES1": lambda v, n0: np.abs(_ar_resid(v, 1)),
                    "ARESL": lambda v, n0: np.abs(_ar_resid(v, 5, n0)),
                    "DRES": lambda v, n0: _devol(v), "ADRES": lambda v, n0: np.abs(_devol(v)),
                    "LARES10": lambda v, n0: np.log(np.abs(_ar_resid(v, 10)) + 1e-3 * v.std()),   # variance break -> level shift
                    "ARES20": lambda v, n0: np.abs(_ar_resid(v, 20)),
                    "RRES10": lambda v, n0: ss.rankdata(np.abs(_ar_resid(v, 10))) / len(v),
                    "RESL10": lambda v, n0: _ar_resid(v, 10, n0),               # AR(10) fitted on before only
                    "ARESL10": lambda v, n0: np.abs(_ar_resid(v, 10, n0)),
                    "RESR10": lambda v, n0: _ar_resid(v, 10, fit_from=n0),      # AR(10) fitted on after only
                    "ARESR10": lambda v, n0: np.abs(_ar_resid(v, 10, fit_from=n0)),
                    "HP": lambda v, n0: v - pd.Series(v).rolling(15, center=True, min_periods=1).median().to_numpy(),
                    "AHP": lambda v, n0: np.abs(v - pd.Series(v).rolling(15, center=True, min_periods=1).median().to_numpy()),
                    # spike-robust versions (error analysis: false alarms are driven by isolated spikes after the boundary)
                    "HAMP": lambda v, n0: _hampel(v),
                    "DHAMP": lambda v, n0: np.diff(_hampel(v), prepend=v[0]),
                    "HAMP3": lambda v, n0: _hampel(v, 9, 3.0),
                    "RESH10": lambda v, n0: _ar_resid(_hampel(v), 10),
                    "ARESH10": lambda v, n0: np.abs(_ar_resid(_hampel(v), 10)),
                    "CSHAMP": lambda v, n0: np.cumsum(_hampel(v))}       # ranks of |residual|                     # squared residual


def expand(prefix, d, out):
    for k, v in d.items():
        if isinstance(v, tuple):
            l, r, w = (np.nan if u is None else float(u) for u in v)
            n = f"{prefix}_{k}"
            out[n + "_l"], out[n + "_r"], out[n + "_w"] = l, r, w
            out[n + "_diff"], out[n + "_ratio"] = r - l, r / (l + 1e-6)
            out[n + "_cl"], out[n + "_cr"] = l / (l + r + 1e-6), r / (l + r + 1e-6)
            out[n + "_rwl"], out[n + "_rwr"] = l / (w + 1e-6), r / (w + 1e-6)
        else:
            out[f"{prefix}_{k}"] = float(v)


def run_chunk(fams, items, trs=TRANSFORMS):
    import warnings
    warnings.filterwarnings("ignore")
    rows = {}
    with np.errstate(all="ignore"):
        for sid, v, n0 in items:
            out = {}
            for fam in fams:
                for tn, tf in trs.items():
                    try:
                        expand(f"{fam}_{tn}", REG[fam](tf(v, n0), n0), out)
                    except Exception:
                        pass
            rows[sid] = out
    return rows


def load_series():
    items = []
    for part in ("train", "test"):
        d = pd.read_parquet(f"data/{part}_series.parquet").sort_values(["id", "time"])
        for sid, g in d.groupby("id", sort=False):
            items.append((sid, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())))
    return items


if __name__ == "__main__":
    ALL = {**TRANSFORMS, **EXTRA_TRANSFORMS}
    fams = [a for a in sys.argv[1:] if a not in ALL]
    trs = {a: ALL[a] for a in sys.argv[1:] if a in ALL} or TRANSFORMS
    suffix = "" if trs is TRANSFORMS else "_" + "_".join(trs)
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_series()
    for fam in fams:
        t = time.time()
        chunks = [items[i::120] for i in range(120)]
        res = Parallel(n_jobs=int(os.environ.get("NTHREADS", 30)))(delayed(run_chunk)([fam], c, trs) for c in chunks)
        D = pd.DataFrame({k: v for r in res for k, v in r.items()}).T.astype(np.float32)
        D = D.replace([np.inf, -np.inf], np.nan)
        D = D.loc[:, D.nunique() > 1]
        D.to_parquet(OUT / f"{fam}{suffix}.parquet")
        print(f"{fam}{suffix}: {D.shape} in {time.time() - t:.0f}s", flush=True)
