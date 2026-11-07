"""Measure each break precisely with the series' own pre-boundary formula (stationary pre: WN / AR1 / AR2, ids and
classes cached in outputs/probe/break_size.parquet). One regression per series:
    x_t = c + d * 1[t >= n0] + p1 x_{t-1} + p2 x_{t-2} + e_t
  level  = d / (1 - p1 - p2) / sd(e before)      long-run level shift in noise-sd units
  spread = sd(e after) / sd(e before)             (same coefficients both sides)
  dep    = AR(1) coefficient after - before       (separate AR(1) fits)
Fine histograms for break vs no-break series.

    python probe/break_measure.py
"""
import time

import numpy as np
import pandas as pd


def measure(v, n0):
    x = v
    n = len(x)
    X = np.column_stack([np.ones(n - 2), (np.arange(2, n) >= n0).astype(float), x[1:-1], x[:-2]])
    b, *_ = np.linalg.lstsq(X, x[2:], rcond=None)
    e = x[2:] - X @ b
    k = n0 - 2
    s0 = e[:k].std()
    ar1 = lambda z: np.polyfit(z[:-1], z[1:], 1)[0]
    return b[1] / (1 - b[2] - b[3]) / s0, e[k:].std() / s0, ar1(x[n0:]) - ar1(x[:n0])


if __name__ == "__main__":
    t0 = time.time()
    C = pd.read_parquet("outputs/probe/break_size.parquet")[["cls", "y"]]
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(C.index))].sort_values(["id", "time"])
    rows = {k: measure(g["value"].to_numpy(), int((g["period"] == 0).sum())) for k, g in S.groupby("id")}
    M = pd.DataFrame(rows, index=["level", "spread", "dep"]).T.join(C)
    M.to_parquet("outputs/probe/break_measure.parquet")
    pd.set_option("display.width", 250)
    for c, bins in [("level", np.r_[-9, np.arange(-3, 3.01, 0.25), 9]),
                    ("spread", np.r_[0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 9]),
                    ("dep", np.r_[-2, np.arange(-0.6, 0.61, 0.1), 2])]:
        h = lambda z: np.histogram(z, bins)[0] / len(z) * 100
        T = pd.DataFrame({"no-break": h(M.loc[M.y == 0, c]), "break": h(M.loc[M.y == 1, c])}, index=[f"{a:g}" for a in bins[:-1]])
        T["excess"] = T["break"] - T["no-break"]
        print(f"{c} (% per bin, bin lower edge)"); print(T.round(1).T.to_string())
    print(f"total {time.time() - t0:.1f}s")
