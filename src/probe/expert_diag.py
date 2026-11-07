"""Box-Jenkins / tsfeatures-style diagnostics: unmatched 9% of the correlated third vs series matched by AR1.
Each row is one standard question an analyst asks when identifying a time-series model.

    python probe/expert_diag.py
"""
import importlib.util
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0], "--n", "1"]
spec = importlib.util.spec_from_file_location("em", "probe/exact_match.py")
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)


def pacf(x, nlags):
    """Durbin-Levinson partial autocorrelations."""
    x = x - x.mean()
    r = np.array([(x[k:] @ x[:len(x) - k]) / (x @ x) for k in range(nlags + 1)])
    phi, out = np.zeros(0), [1.0]
    for k in range(1, nlags + 1):
        a = (r[k] - phi @ r[1:k][::-1]) / (1 - phi @ r[1:k])
        phi = np.r_[phi - a * phi[::-1], a]
        out.append(a)
    return np.array(out)


def kpss(x):
    """KPSS level-stationarity statistic with Newey-West long-run variance (lag 4*(n/100)^0.25)."""
    n = len(x)
    e = x - x.mean()
    L = int(4 * (n / 100) ** 0.25)
    lr = e @ e / n + 2 * sum((1 - j / (L + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, L + 1))
    return (np.cumsum(e) ** 2).sum() / n ** 2 / lr


def diag(v):
    n, t = len(v), np.arange(len(v))
    x = (v - v.mean()) / v.std()
    sl = stats.linregress(t, x)
    w = n // 10
    m = np.array([x[i:i + w].mean() for i in range(0, n - w + 1, w)])
    s = np.array([x[i:i + w].var() for i in range(0, n - w + 1, w)])
    r = np.convolve(x, np.ones(w) / w, "valid")
    pw = np.abs(np.fft.rfft(x)) ** 2
    pw = pw[1:]
    ff = np.fft.rfftfreq(n)[1:]
    pc = pacf(x, nlags=10)
    # GPH long-memory estimate d (log-periodogram on the lowest sqrt(n) frequencies)
    k = int(np.sqrt(n))
    d = -stats.linregress(np.log(4 * np.sin(np.pi * ff[:k]) ** 2), np.log(pw[:k])).slope
    e = np.diff(x)
    return {
        "1 trend: |t| of linear slope": abs(sl.slope / sl.stderr),
        "1 trend: R2 of straight line": sl.rvalue ** 2,
        "2 level: stability (var of window means)": m.var(),
        "2 level: max level shift (adjacent windows)": np.abs(np.diff(r[::w])).max() if len(r[::w]) > 1 else 0,
        "3 variance: lumpiness (var of window vars)": s.var(),
        "4 stationary: KPSS stat (>0.46 = not)": kpss(x),
        "5 memory: GPH d (0 = short, >0.2 = long)": d,
        "6 order: # PACF lags beyond +-2/sqrt(n)": int((np.abs(pc[1:]) > 2 / np.sqrt(n)).sum()),
        "7 period: Fisher g (max share of power)": pw.max() / pw.sum(),
        "7 period: freq of max power": ff[pw.argmax()],
        "8 tails: excess kurtosis of diff": stats.kurtosis(e),
    }


if __name__ == "__main__":
    t0 = time.time()
    R = np.load("outputs/probe/real_acf12_l0.npy")
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    ids0 = np.sort(lab.index[lab == 0].to_numpy())
    keep = ids0[(np.abs(R[:, 0]) >= 0.1) & (R[:, 0] <= 0.9)]
    ids = set(np.random.default_rng(0).choice(keep, 2000, replace=False))
    S = pd.read_parquet("data/train_series.parquet", columns=["id", "time", "value"])
    S = S[S.id.isin(ids)].sort_values(["id", "time"])
    V = [g.to_numpy() for _, g in S.groupby("id")["value"]]
    fits = Parallel(n_jobs=30)(delayed(em.one)(v) for v in V)
    grp = np.array(["unmatched" if not w.startswith("SIN") and not any(ok.values()) else
                    "AR1 matched" if w == "AR1" and ok["AR1"] else "other" for w, ok in fits])
    D = pd.DataFrame(Parallel(n_jobs=30)(delayed(diag)(v) for v in V))
    D["grp"] = grp
    T = D[D.grp != "other"].groupby("grp").median().T
    T["ratio"] = T["unmatched"] / T["AR1 matched"]
    pd.set_option("display.width", 200)
    print(f"medians (unmatched n={sum(grp == 'unmatched')}, AR1 matched n={sum(grp == 'AR1 matched')})")
    print(T.round(3).to_string())
    print(f"total {time.time() - t0:.1f}s")
