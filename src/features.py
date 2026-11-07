"""Per-series before/after features for KU-SBB structural break detection.

Input long format: id, time, value, period (0 = before boundary, 1 = after).
Every feature compares the period-0 segment with the period-1 segment of one series,
so nothing is fit across series and there is no leakage between CV folds.
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _acf(x, lag):
    if len(x) <= lag + 2:
        return np.nan
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _seg_stats(x):
    """Summary statistics of one segment."""
    mad = np.median(np.abs(x - np.median(x)))
    dx = np.diff(x) if len(x) > 1 else np.array([0.0])
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    lo, hi = np.quantile(x, [0.1, 0.9])
    trimmed = x[(x >= lo) & (x <= hi)]
    return {
        "mean": x.mean(),
        "std": x.std(),
        "median": q[2],
        "mad": mad,
        "iqr": q[3] - q[1],
        "q05": q[0],
        "q95": q[4],
        "tmean": trimmed.mean() if len(trimmed) else x.mean(),
        "skew": stats.skew(x) if len(x) > 2 else 0.0,
        "kurt": stats.kurtosis(x) if len(x) > 3 else 0.0,
        "acf1": _acf(x, 1),
        "acf2": _acf(x, 2),
        "dstd": dx.std(),
        "dabs": np.abs(dx).mean(),
        "slope": np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 2 else 0.0,
        "zero_cross": np.mean(np.diff(np.sign(x - x.mean())) != 0) if len(x) > 1 else 0.0,
    }


def _spectral(x, nbands=4):
    if len(x) < 8:
        return np.full(nbands, np.nan)
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    p = p[1:]
    tot = p.sum() + 1e-12
    return np.array([b.sum() / tot for b in np.array_split(p, nbands)])


def series_features(sid, values, periods):
    x0 = values[periods == 0]
    x1 = values[periods == 1]
    f = {"id": sid, "n0": len(x0), "n1": len(x1)}
    if len(x0) < 3 or len(x1) < 3:
        return f

    # Scale by the "before" segment so features are comparable across series.
    scale = x0.std() + 1e-8
    s0, s1 = _seg_stats(x0), _seg_stats(x1)
    for k in s0:
        f[f"b_{k}"] = s0[k]
        f[f"a_{k}"] = s1[k]
    for k in ("mean", "median", "tmean", "q05", "q95"):
        f[f"d_{k}"] = (s1[k] - s0[k]) / scale
    for k in ("std", "mad", "iqr", "dstd", "dabs"):
        f[f"lr_{k}"] = np.log((s1[k] + 1e-8) / (s0[k] + 1e-8))
    for k in ("skew", "kurt", "acf1", "acf2", "slope", "zero_cross"):
        f[f"d_{k}"] = s1[k] - s0[k]

    # Two-sample tests: location, scale, distribution.
    f["t_stat"], f["t_p"] = stats.ttest_ind(x0, x1, equal_var=False)
    f["mw_p"] = stats.mannwhitneyu(x0, x1).pvalue
    ks = stats.ks_2samp(x0, x1)
    f["ks_stat"], f["ks_p"] = ks.statistic, ks.pvalue
    f["lev_p"] = stats.levene(x0, x1, center="median").pvalue
    f["ansari_p"] = stats.ansari(x0, x1).pvalue
    f["wass"] = stats.wasserstein_distance(x0 / scale, x1 / scale)
    f["energy"] = stats.energy_distance(x0 / scale, x1 / scale)
    for k in ("t_p", "mw_p", "ks_p", "lev_p", "ansari_p"):
        f[f"nl_{k}"] = -np.log10(max(f[k], 1e-300))

    # Same tests on first differences (catches dynamics changes, not level shifts).
    d0, d1 = np.diff(x0), np.diff(x1)
    f["dks_stat"] = stats.ks_2samp(d0, d1).statistic
    f["nl_dlev_p"] = -np.log10(max(stats.levene(d0, d1, center="median").pvalue, 1e-300))

    # Local contrast right at the boundary vs. far from it (persistent vs. temporary).
    w = max(5, min(len(x0), len(x1)) // 5)
    f["d_mean_near"] = (x1[:w].mean() - x0[-w:].mean()) / scale
    f["d_mean_far"] = (x1[-w:].mean() - x0[-w:].mean()) / scale
    f["persist"] = f["d_mean_far"] * np.sign(f["d_mean_near"]) if f["d_mean_near"] != 0 else 0.0

    # CUSUM on the full series standardised by the before segment.
    z = (np.concatenate([x0, x1]) - x0.mean()) / scale
    cs = np.cumsum(z - z.mean())
    f["cusum_max"] = np.abs(cs).max() / np.sqrt(len(z))
    f["cusum_at_boundary"] = np.abs(cs[len(x0) - 1]) / np.sqrt(len(z))

    sp0, sp1 = _spectral(x0), _spectral(x1)
    for i in range(len(sp0)):
        f[f"spec{i}_d"] = sp1[i] - sp0[i]
    f.update(extra_features(x0, x1))
    f.update(adia_features(x0, x1))
    return f


# ---- Ideas from public ADIA structural-break solutions (see README "References") ----

def _tests(a, b, tag):
    """Compact test battery used on every transform (farukcan-saglam / leo-nguyen style)."""
    f = {}
    if len(a) < 5 or len(b) < 5:
        return f
    s = np.std(np.concatenate([a, b])) + 1e-8
    f[f"{tag}_dm"] = (b.mean() - a.mean()) / s
    f[f"{tag}_lsr"] = np.log((b.std() + 1e-8) / (a.std() + 1e-8))
    f[f"{tag}_ks"] = stats.ks_2samp(a, b).statistic
    f[f"{tag}_mw"] = stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 0.5
    f[f"{tag}_lev"] = -np.log10(max(stats.levene(a, b, center="median").pvalue, 1e-300))
    f[f"{tag}_wass"] = stats.wasserstein_distance(a / s, b / s)
    return f


def _perm_entropy(x, m=4):
    if len(x) < m + 10:
        return np.nan
    pats = np.argsort(np.lib.stride_tricks.sliding_window_view(x, m), axis=1)
    codes = (pats * (m ** np.arange(m))).sum(1)
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return -(p * np.log(p)).sum() / np.log(float(np.prod(range(1, m + 1))))


def _compress_ratio(x, lo, hi):
    import zlib
    q = np.clip(((x - lo) / (hi - lo + 1e-12) * 255), 0, 255).astype(np.uint8).tobytes()
    return len(zlib.compress(q, 9)) / max(len(q), 1)


def _haar_energy(x, levels=5):
    e = []
    a = x - x.mean()
    for _ in range(levels):
        if len(a) < 4:
            e.append(np.nan)
            continue
        a = a[: len(a) // 2 * 2]
        d = (a[0::2] - a[1::2]) / np.sqrt(2)
        a = (a[0::2] + a[1::2]) / np.sqrt(2)
        e.append(np.mean(d ** 2))
    return np.array(e)


def _ewma_vol(x, lam=0.94):
    r = np.diff(x)
    v = np.empty_like(r)
    v[0] = r[:20].var() + 1e-12
    for t in range(1, len(r)):
        v[t] = lam * v[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(v)


def adia_features(x0, x1):
    f = {}
    xa = np.concatenate([x0, x1])
    n0 = len(x0)

    # Boundary windows (leo-nguyen): the same tests on +-w readings around the boundary.
    for w in (7, 15, 30, 100):
        if n0 >= w and len(x1) >= w:
            f.update(_tests(x0[-w:], x1[:w], f"bw{w}"))

    # Transforms x tests (farukcan-saglam): ranks are pooled, rolling stats use window 10.
    rk = stats.rankdata(xa) / len(xa)
    dx = np.diff(xa)
    ad = np.abs(xa - np.median(x0))
    k = 10
    rm = np.convolve(xa, np.ones(k) / k, "valid")
    rv = np.sqrt(np.maximum(np.convolve(xa ** 2, np.ones(k) / k, "valid") - rm ** 2, 0))
    for tag, v, split in (("rank", rk, n0), ("diff", dx, n0 - 1), ("absdev", ad, n0),
                          ("rmean", rm, n0 - k + 1), ("rstd", rv, n0 - k + 1),
                          ("absdiff", np.abs(dx), n0 - 1)):
        # Drop the window entries that straddle the boundary so before/after stay separate.
        gap = k - 1 if tag in ("rmean", "rstd") else 0
        f.update(_tests(v[:split], v[split + gap:], tag))

    # EWMA / GARCH-like volatility (valuable-j).
    vol = _ewma_vol(xa)
    vb, va = vol[: n0 - 1], vol[n0 - 1 + 30:]  # skip the burn-in right after the boundary
    if len(va) > 5:
        f["ewma_lr"] = np.log((np.median(va) + 1e-8) / (np.median(vb[20:]) + 1e-8))
        f["ewma_out"] = np.mean(va > np.quantile(vb[20:], 0.95))
    rr = np.diff(xa) / (vol + 1e-8)  # volatility-standardised innovations
    f.update(_tests(rr[: n0 - 1], rr[n0 - 1:], "stdres"))

    # Complexity (valuable-j): permutation entropy and zlib compression on a shared scale.
    for m in (3, 5):
        f[f"pe{m}_b"] = _perm_entropy(x0, m)
        f[f"pe{m}_d"] = _perm_entropy(x1, m) - f[f"pe{m}_b"]
    lo, hi = np.quantile(xa, [0.01, 0.99])
    c0, c1 = _compress_ratio(x0, lo, hi), _compress_ratio(x1, lo, hi)
    f["zlib_b"], f["zlib_d"] = c0, c1 - c0
    d0, d1 = np.diff(x0), np.diff(x1)
    lo, hi = np.quantile(np.concatenate([d0, d1]), [0.01, 0.99])
    f["zlib_diff_d"] = _compress_ratio(d1, lo, hi) - _compress_ratio(d0, lo, hi)

    # Wavelet energy by scale (leo-nguyen used db4; Haar keeps this dependency-free).
    e0, e1 = _haar_energy(x0), _haar_energy(x1)
    for i in range(len(e0)):
        f[f"haar{i}_lr"] = np.log((e1[i] + 1e-10) / (e0[i] + 1e-10))
    f["haar_shape_l1"] = np.nansum(np.abs(e1 / np.nansum(e1) - e0 / np.nansum(e0)))

    # Coefficient of variation per segment (valuable-j's "magic" feature, per segment here).
    f["cv_b"] = x0.std() / (abs(x0.mean()) + 1e-8)
    f["lcv_d"] = np.log((x1.std() / (abs(x1.mean()) + 1e-8) + 1e-8) / (f["cv_b"] + 1e-8))
    return f


def _ar_fit(x, p):
    """Least-squares AR(p) with intercept: returns coefficients, residual sum of squares, n used."""
    X = np.column_stack([np.ones(len(x) - p)] + [x[p - k - 1:len(x) - k - 1] for k in range(p)])
    yv = x[p:]
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r = yv - X @ beta
    return beta, float(r @ r), len(yv), r


def extra_features(x0, x1, p=3):
    """Robust, autoregressive, volatility and persistence contrasts."""
    f = {}
    med0 = np.median(x0)
    rs0 = 1.4826 * np.median(np.abs(x0 - med0)) + 1e-8

    # Spikes: fraction of outliers on each side, then contrasts on winsorised data.
    f["spike_b"] = np.mean(np.abs(x0 - med0) > 5 * rs0)
    f["spike_a"] = np.mean(np.abs(x1 - med0) > 5 * rs0)
    med1 = np.median(x1)
    rs1 = 1.4826 * np.median(np.abs(x1 - med1)) + 1e-8
    w0 = np.clip(x0, med0 - 5 * rs0, med0 + 5 * rs0)
    w1 = np.clip(x1, med1 - 5 * rs1, med1 + 5 * rs1)
    f["w_dmean"] = (w1.mean() - w0.mean()) / rs0
    f["w_lrstd"] = np.log((w1.std() + 1e-8) / (w0.std() + 1e-8))
    f["w_acf1_d"] = _acf(w1, 1) - _acf(w0, 1)
    f["r_dmed"] = (med1 - med0) / rs0
    f["r_lrmad"] = np.log(rs1 / rs0)
    f["r_zmed"] = f["r_dmed"] / np.sqrt(1 / len(x0) + 1 / len(x1))

    # Gaussian likelihood-ratio for a change in mean and/or variance.
    xa = np.concatenate([x0, x1])
    n0, n1, n = len(x0), len(x1), len(xa)
    v0, v1, va = x0.var() + 1e-12, x1.var() + 1e-12, xa.var() + 1e-12
    f["glr"] = (n * np.log(va) - n0 * np.log(v0) - n1 * np.log(v1)) / n

    # AR(p) per segment and pooled: coefficient changes and a Chow-type F statistic.
    b0, rss0, m0, e0 = _ar_fit(x0, p)
    b1, rss1, m1, e1 = _ar_fit(x1, p)
    _, rssp, _, _ = _ar_fit(xa, p)
    k = p + 1
    f["chow_f"] = np.log1p(max((rssp - rss0 - rss1) / k, 0) / ((rss0 + rss1) / max(m0 + m1 - 2 * k, 1) + 1e-12))
    for i in range(1, k):
        f[f"ar{i}_b"] = b0[i]
        f[f"ar{i}_d"] = b1[i] - b0[i]
    f["ar_lrres"] = 0.5 * np.log((rss1 / m1 + 1e-12) / (rss0 / m0 + 1e-12))
    f["ar_lrres_robust"] = np.log((np.median(np.abs(e1)) + 1e-8) / (np.median(np.abs(e0)) + 1e-8))
    f["ar_res_kurt_d"] = stats.kurtosis(e1) - stats.kurtosis(e0)

    # Autocorrelation profiles of levels, absolute deviations (volatility clustering) and squares.
    for lag in (3, 5, 10):
        f[f"acf{lag}_d"] = _acf(x1, lag) - _acf(x0, lag)
    a0, a1 = np.abs(x0 - x0.mean()), np.abs(x1 - x1.mean())
    for lag in (1, 2, 5):
        f[f"vacf{lag}_b"] = _acf(a0, lag)
        f[f"vacf{lag}_d"] = _acf(a1, lag) - _acf(a0, lag)

    # Variance ratios (random-walk / mean-reversion behaviour).
    for q in (2, 4, 8):
        def vr(x):
            if len(x) <= 2 * q:
                return np.nan
            return np.var(x[q:] - x[:-q]) / (q * np.var(np.diff(x)) + 1e-12)
        f[f"vr{q}_b"] = vr(x0)
        f[f"vr{q}_d"] = vr(x1) - vr(x0)

    # Finer log-ratio spectrum.
    s0, s1 = _spectral(x0, 8), _spectral(x1, 8)
    for i in range(8):
        f[f"lspec{i}"] = np.log((s1[i] + 1e-6) / (s0[i] + 1e-6))

    # Persistence: is the shift present in every third of the after segment,
    # and how large is it relative to natural drift between thirds of the before segment?
    t0, t1 = np.array_split(x0, 3), np.array_split(x1, 3)
    m_b = np.array([t.mean() for t in t0])
    m_a = np.array([t.mean() for t in t1])
    sh = (m_a - x0.mean()) / rs0
    f["persist_min"] = np.min(np.abs(sh)) * (1 if np.all(np.sign(sh) == np.sign(sh[0])) else 0)
    f["persist_med"] = np.median(sh)
    drift = np.ptp(m_b) / rs0 + 1e-3
    f["shift_vs_drift"] = np.abs(np.median(sh)) / drift
    sd_b = np.array([t.std() for t in t0]) + 1e-8
    sd_a = np.array([t.std() for t in t1]) + 1e-8
    lr = np.log(sd_a / sd_b.mean())
    f["vpersist_min"] = np.min(np.abs(lr)) * (1 if np.all(np.sign(lr) == np.sign(lr[0])) else 0)
    f["vshift_vs_drift"] = np.abs(np.median(lr)) / (np.ptp(np.log(sd_b)) + 1e-2)

    # Rank tests on dynamics.
    f["nl_mw_absdiff"] = -np.log10(max(stats.mannwhitneyu(np.abs(np.diff(x0)), np.abs(np.diff(x1))).pvalue, 1e-300))
    f["ks_absdev"] = stats.ks_2samp(a0 / (a0.mean() + 1e-8), a1 / (a1.mean() + 1e-8)).statistic
    f.update(dependence_features(x0, x1, w0, w1))
    return f


def _acf_vec(x, L):
    x = x - x.mean()
    d = x @ x + 1e-12
    return np.array([x[:-l] @ x[l:] / d for l in range(1, L + 1)])


def _cum_periodogram(x, grid):
    p = np.abs(np.fft.rfft(x - x.mean()))[1:] ** 2
    c = np.cumsum(p) / (p.sum() + 1e-12)
    return np.interp(grid, np.linspace(0, 1, len(c)), c)


def _rolling(x, w):
    n = len(x) // w
    b = x[: n * w].reshape(n, w)
    m = b.mean(1)
    s = b.std(1) + 1e-8
    c = b - m[:, None]
    r1 = (c[:, :-1] * c[:, 1:]).sum(1) / ((c * c).sum(1) + 1e-12)
    return m, np.log(s), r1


def dependence_features(x0, x1, w0, w1, L=20):
    """Changes in serial dependence, scale tests and window-level persistence."""
    f = {}
    L = min(L, len(x0) // 4, len(x1) // 4)
    for tag, a, b in (("", x0, x1), ("w", w0, w1)):
        r0, r1 = _acf_vec(a, L), _acf_vec(b, L)
        h = 1 / len(a) + 1 / len(b)
        f[f"{tag}acf_port"] = np.sum((r1 - r0) ** 2) / h / L
        f[f"{tag}acf_port5"] = np.sum((r1[:5] - r0[:5]) ** 2) / h / 5
        f[f"{tag}acf_sumd"] = np.sum(r1[:5] - r0[:5])
    # Rank autocorrelation, immune to single spikes.
    sp = lambda x: stats.spearmanr(x[:-1], x[1:]).statistic
    f["sacf1_b"] = sp(x0)
    f["sacf1_d"] = sp(x1) - f["sacf1_b"]
    sp2 = lambda x: stats.spearmanr(x[:-2], x[2:]).statistic
    f["sacf2_d"] = sp2(x1) - sp2(x0)
    # Spectral shape: KS / L1 distance between cumulative periodograms.
    grid = np.linspace(0, 1, 64)
    c0, c1 = _cum_periodogram(x0, grid), _cum_periodogram(x1, grid)
    f["cpg_ks"] = np.max(np.abs(c1 - c0))
    f["cpg_l1"] = np.mean(np.abs(c1 - c0))
    f["cpg_ks_z"] = f["cpg_ks"] / np.sqrt(1 / len(x0) + 1 / len(x1))
    c0w, c1w = _cum_periodogram(w0, grid), _cum_periodogram(w1, grid)
    f["wcpg_ks"] = np.max(np.abs(c1w - c0w))
    # Scale and distribution tests.
    f["nl_fligner"] = -np.log10(max(stats.fligner(x0, x1).pvalue, 1e-300))
    f["nl_mood"] = -np.log10(max(stats.mood(x0, x1).pvalue, 1e-300))
    f["cvm"] = stats.cramervonmises_2samp(x0, x1).statistic
    f["ad"] = stats.anderson_ksamp([x0, x1], method=None).statistic if hasattr(stats, "anderson_ksamp") else np.nan
    # Window-level contrasts: where does the median after-window fall in the before-window distribution?
    for w in (16, 32):
        if len(x0) < 3 * w or len(x1) < 3 * w:
            continue
        mb, sb, rb = _rolling(x0, w)
        ma, sa, ra = _rolling(x1, w)
        for k, b, a in (("m", mb, ma), ("s", sb, sa), ("r", rb, ra)):
            spread = np.std(b) + 1e-6
            f[f"win{w}_{k}_z"] = (np.median(a) - np.median(b)) / spread
            f[f"win{w}_{k}_out"] = np.mean((a > b.max()) | (a < b.min()))
            f[f"win{w}_{k}_mw"] = stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 0.5
    return f


def build_features(df, n_jobs=-1):
    df = df.sort_values(["id", "time"], kind="stable")
    groups = [(sid, g["value"].to_numpy(float), g["period"].to_numpy(int)) for sid, g in df.groupby("id", sort=False)]
    rows = Parallel(n_jobs=n_jobs, batch_size=256)(delayed(series_features)(*g) for g in groups)
    return pd.DataFrame(rows).set_index("id")
