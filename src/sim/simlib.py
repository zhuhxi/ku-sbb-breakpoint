"""KU-SBB simulator: regenerate real no-break series and inject breaks.

Donors are the real label-0 train series of the training rows passed in (so a CV fold never
sees donors from its own validation rows). For each donor:
  * stationary donor  (AR(5) coefficient sum < 0.97): x_t = c + d t/n + sum phi_i x_{t-i} + e_t on the
    levels, fitted jointly; only a significant trend and significant lags are kept (fit_sparse);
  * unit-root donor   (sum >= 0.97): the same AR(5) on the differences, then cumulated.
Innovations are a smoothed bootstrap of the donor's own residuals (keeps the donor's tails). Lengths n0, n1 are the donor's.

A break at t = n0 (label 1) is one of:
  mean  x[n0:] += delta * sd(x)                       delta ~ N(0, s_mean)
  var   e[n0:] *= exp(g)                              g ~ N(0, s_var)
  ar    phi -> phi + eta * (unit vector)              eta ~ N(0, s_ar), kept stable
  dist  e[n0:] := residuals of another donor, rescaled to the same sd
Type probabilities and magnitudes are the calibration knobs (see calibrate.py).
"""
import numpy as np
import pandas as pd

P = 5
JITTER = 0.2
VOL_LB, VOL_WIN = 25.0, 21
KNOBS = {"p": [0.3, 0.5, 0.1, 0.1], "s_mean": 1.1, "s_var": 0.31, "s_ar": 0.4}   # mean, var, ar, dist


def fit_ar(x, p=P, trend=False):
    cols = [np.ones(len(x) - p)] + ([np.arange(p, len(x)) / len(x)] if trend else [])
    X = np.column_stack(cols + [x[p - i:-i] for i in range(1, p + 1)])
    b, *_ = np.linalg.lstsq(X, x[p:], rcond=None)
    return b, x[p:] - X @ b


def fit_sparse(z, trend, t_trend=6.0, t_lag=3.0, p=P):
    """OLS of z_t on [1, t/n, z_{t-1..p}], then refit keeping only the trend if |t| > t_trend and the
    lags with |t| >= t_lag. Copying every noisy estimate would give each simulated series spurious
    trend and dependence (it showed up as too-large before/after differences)."""
    cols = [np.ones(len(z) - p), np.arange(p, len(z)) / len(z)] + [z[p - i:-i] for i in range(1, p + 1)]
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, z[p:], rcond=None)
    e = z[p:] - X @ b
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (e @ e) / (len(e) - X.shape[1]))
    t = np.abs(b) / (se + 1e-300)
    keep = np.r_[True, trend and t[1] > t_trend, t[2:] >= t_lag]
    bk, *_ = np.linalg.lstsq(X[:, keep], z[p:], rcond=None)
    full = np.zeros(len(b))
    full[keep] = bk
    return full[0], full[1], full[2:], z[p:] - X[:, keep] @ bk


def make_donors(series):
    """series: list of (x, n0) for real no-break series. Returns a list of donor dicts."""
    out = []
    for x, n0 in series:
        b, _ = fit_ar(x, trend=True)
        unit = b[2:].sum() >= 0.97
        c, d, phi, e = fit_sparse(np.diff(x), False) if unit else fit_sparse(x, True)
        e = e - e.mean()
        # Volatility clustering: if the squared residuals are clearly autocorrelated (Ljung-Box on lags
        # 1-5), keep the donor's smoothed volatility path and bootstrap the standardized residuals.
        s2 = e ** 2 - (e ** 2).mean()
        ac = np.array([s2[:-k] @ s2[k:] for k in range(1, 6)]) / (s2 @ s2 + 1e-300)
        v = np.ones(len(e))
        if len(e) * (ac ** 2).sum() > VOL_LB:
            v = np.sqrt(pd.Series(e ** 2).rolling(VOL_WIN, center=True, min_periods=1).mean().to_numpy())
            v = v / np.sqrt(np.mean(v ** 2))
        out.append({"c": c, "d": d, "phi": phi, "e": e / v, "v": v, "unit": unit, "x0": x[:P + 1].copy(),
                    "sd": x.std(), "n": len(x), "n0": n0})
    return out


def _boot(e, n, rng):
    """Smoothed iid bootstrap: resample residuals and add a little kernel noise. Copying exact values
    or blocks repeats ordinal patterns, which real series never do (it lowered permutation entropy)."""
    out = e[rng.integers(0, len(e), n)]
    return (out + rng.normal(0, JITTER * e.std(), n)) / np.sqrt(1 + JITTER ** 2)


def _stable(phi):
    r = np.abs(np.roots(np.r_[1, -phi]))
    return r.max() < 0.995 if len(r) else True


def _run(c, phi, e, init, d=0.0, n=1, t0=0):
    """x_t = c + d * (t / n) + sum phi_i x_{t-i} + e_t, with t counted from t0 for the first innovation."""
    p = len(phi)
    x = np.empty(len(e) + p)
    x[:p] = init
    for t in range(p, len(x)):
        x[t] = c + d * (t0 + t - p) / n + phi @ x[t - p:t][::-1] + e[t - p]
    return x[p:]


def simulate_one(don, n0, n1, label, knobs, rng, donors):
    n = n0 + n1
    burn = 100
    e = _boot(don["e"], n + burn, rng) * np.r_[np.full(n + burn, don["v"][0]), don["v"]][-(n + burn):]   # path aligned to the end
    c, d, phi = don["c"], don["d"], don["phi"].copy()
    btype = None
    if label:
        btype = rng.choice(["mean", "var", "ar", "dist"], p=knobs["p"])
    k = burn + n0
    if btype == "var":
        e[k:] *= np.exp(rng.normal(0, knobs["s_var"]))
    elif btype == "dist":
        o = donors[rng.integers(len(donors))]["e"]
        o = _boot(o, len(e) - k, rng)
        e[k:] = o / (o.std() + 1e-12) * e[k:].std()
    init = don["x0"][:P] if not don["unit"] else np.diff(don["x0"])[:P]
    if btype == "ar":
        phi2 = phi.copy()
        for _ in range(20):
            cand = phi + rng.normal(0, knobs["s_ar"]) * np.eye(P)[0]
            if _stable(cand) or don["unit"]:
                phi2 = cand
                break
        x = np.r_[_run(c, phi, e[:k], init, d, n, -burn)]
        x = np.r_[x, _run(c, phi2, e[k:], x[-P:], d, n, n0)]
    else:
        x = _run(c, phi, e, init, d, n, -burn)
    if don["unit"]:
        x = np.cumsum(x) + don["x0"][0]
    x = x[burn:]
    if not np.isfinite(x).all() or np.abs(x).max() > 1e4:
        return None, None
    if btype == "mean":
        x[n0:] += rng.normal(0, knobs["s_mean"]) * x[:n0].std()
    return x, btype


def simulate(donors, n_series, knobs, seed=0, prefix="sim"):
    """Returns (long DataFrame id/time/value/period, labels Series, break-type Series)."""
    rng = np.random.default_rng(seed)
    rows, labs, types = [], {}, {}
    i = 0
    while len(labs) < n_series:
        don = donors[rng.integers(len(donors))]
        n0 = don["n0"]
        n1 = don["n"] - n0
        label = int(rng.random() < 0.5)
        x, bt = simulate_one(don, n0, n1, label, knobs, rng, donors)
        i += 1
        if x is None:
            continue
        sid = f"{prefix}{seed}_{len(labs)}"
        rows.append(pd.DataFrame({"id": sid, "time": np.arange(len(x)), "value": x,
                                  "period": np.r_[np.zeros(n0, np.int8), np.ones(n1, np.int8)]}))
        labs[sid], types[sid] = label, bt or "none"
    return pd.concat(rows, ignore_index=True), pd.Series(labs, name="label"), pd.Series(types, name="btype")
