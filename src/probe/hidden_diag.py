"""Expert checklist on the 'hidden' breaks: break series whose level / spread / lag-1 dependence changes all lie inside
the no-break range (below the no-break 90% quantile, same pre-boundary structure). For each extra before-vs-after
quantity, compare the share of hidden breaks beyond the no-break 95% quantile (5% expected if nothing changed).

    python probe/hidden_diag.py --n 4000
"""
import argparse
import importlib.util
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4000)
args = ap.parse_args()
sys.argv = [sys.argv[0], "x"]
spec = importlib.util.spec_from_file_location("cp", "probe/calib_params.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def ac(x, k):
    x = x - x.mean()
    return (x[k:] @ x[:-k]) / (x @ x)


def ar2_resid(x):
    X = np.column_stack([np.ones(len(x) - 2), x[1:-1], x[:-2]])
    b, *_ = np.linalg.lstsq(X, x[2:], rcond=None)
    return x[2:] - X @ b, b


def one(v, n0):
    a, b = v[:n0], v[n0:]
    ea, ba = ar2_resid(a)
    eb, bb = ar2_resid(b)
    sd = a.std()
    h = len(b) // 2
    pa, pb = np.abs(np.fft.rfft(a - a.mean())) ** 2, np.abs(np.fft.rfft(b - b.mean())) ** 2
    fa, fb = np.fft.rfftfreq(len(a)), np.fft.rfftfreq(len(b))
    ta, tb = stats.linregress(np.arange(len(a)), a).slope, stats.linregress(np.arange(len(b)), b).slope
    return dict(
        cls=cp.classify(a)["cls"],
        # the three official changes (used to pick 'hidden' breaks)
        level=abs(b.mean() - a.mean()) / sd, spread=abs(np.log(eb.std() / ea.std())), dep=abs(ac(b, 1) - ac(a, 1)),
        # expert checklist: side effects / other ways a change can show
        **{"lag2 dep": abs(ac(b, 2) - ac(a, 2)), "lags 3-10 dep": np.abs([ac(b, k) - ac(a, k) for k in range(3, 11)]).sum(),
           "AR2 coef": abs(bb[2] - ba[2]),
           "robust level (median)": abs(np.median(b) - np.median(a)) / sd,
           "robust spread (MAD)": abs(np.log(stats.median_abs_deviation(eb) / stats.median_abs_deviation(ea))),
           "noise shape (kurtosis)": abs(stats.kurtosis(eb) - stats.kurtosis(ea)),
           "noise skew": abs(stats.skew(eb) - stats.skew(ea)),
           "spectral peak freq": abs(fb[1:][pb[1:].argmax()] - fa[1:][pa[1:].argmax()]),
           "trend slope": abs(tb - ta) * len(v) / sd,
           "gradual: level late vs early after": abs(b[h:].mean() - b[:h].mean()) / sd,
           "gradual: spread late vs early after": abs(np.log(eb[h:].std() / eb[:h].std())),
           "local: first 30 after vs last 30 before": abs(b[:30].mean() - a[-30:].mean()) / sd,
           "big spikes after (>4 sd) count": int((np.abs(eb) > 4 * ea.std()).sum()) - int((np.abs(ea) > 4 * ea.std()).sum() * len(eb) / len(ea))})


if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    r = np.random.default_rng(1)
    ids = np.r_[r.choice(lab.index[lab == 0], args.n, False), r.choice(lab.index[lab == 1], args.n, False)]
    S = pd.read_parquet("data/train_series.parquet")
    S = S[S.id.isin(set(ids))].sort_values(["id", "time"])
    G = {k: (g["value"].to_numpy(), int((g["period"] == 0).sum())) for k, g in S.groupby("id")}
    D = pd.DataFrame(Parallel(n_jobs=30)(delayed(one)(v, n0) for v, n0 in G.values()), index=list(G))
    D["y"] = lab.reindex(D.index).to_numpy()
    D = D[D.cls.isin(["WN", "AR1", "AR2"])]
    Q = [c for c in D.columns if c not in ("cls", "y")]
    hid0 = pd.Series(True, index=D.index)
    for c, g in D.groupby("cls"):
        q90 = g.loc[g.y == 0, ["level", "spread", "dep"]].quantile(0.9)
        hid0[g.index] = (g[["level", "spread", "dep"]] < q90).all(axis=1)
    H1, H0 = D[(D.y == 1) & hid0], D[(D.y == 0) & hid0]
    print(f"stationary: break {sum(D.y == 1)}, hidden breaks {len(H1)} ({len(H1) / sum(D.y == 1):.0%}); "
          f"no-break passing the same filter {len(H0)}   ({time.time() - t0:.1f}s)")
    print("share beyond the no-break 95% quantile (same filter); 0.05 = nothing there")
    res = {c: (H1[c] > H0[c].quantile(0.95)).mean() for c in Q if c not in ("level", "spread", "dep")}
    for c, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {c:40s} {v:.3f}")
    H1.to_parquet("outputs/probe/hidden_breaks.parquet")
    pick = np.random.default_rng(2).choice(H1.index, 6, replace=False)
    fig, ax = plt.subplots(6, 1, figsize=(15, 13))
    for i, k in enumerate(pick):
        v, n0 = G[k]
        w = 30
        rm = pd.Series(v).rolling(w, center=True).mean()
        rs = pd.Series(np.r_[0, np.diff(v)]).rolling(w, center=True).std()
        ax[i].plot(v, lw=0.5, color="0.45")
        ax[i].plot(rm, color="tab:red", lw=1.5, label="rolling mean (30)")
        ax[i].plot(rm.mean() + 2 * rs, color="tab:green", lw=1, label="rolling spread (30)")
        ax[i].axvline(n0, color="tab:blue", ls="--")
        ax[i].set_title(f"HIDDEN break  id={k}  pre-structure={H1.loc[k, 'cls']}  n={len(v)}", fontsize=9, loc="left")
    ax[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig("outputs/probe/hidden_examples.png", dpi=80)
    print(f"total {time.time() - t0:.1f}s")
