"""Plot series of the correlated third that no simple formula matched, next to ones matched by AR1 (raw series,
autocorrelation up to lag 40, and spectrum).

    python probe/plot_unmatched.py
"""
import importlib.util
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0], "--n", "1"]
spec = importlib.util.spec_from_file_location("em", "probe/exact_match.py")
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)


def acf(x, h=40):
    x = x - x.mean()
    return np.array([(x[k:] @ x[:len(x) - k]) / (x @ x) for k in range(1, h + 1)])


R = np.load("outputs/probe/real_acf12_l0.npy")
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
ids0 = np.sort(lab.index[lab == 0].to_numpy())
keep = ids0[(np.abs(R[:, 0]) >= 0.1) & (R[:, 0] <= 0.9)]
ids = set(np.random.default_rng(0).choice(keep, 2000, replace=False))
S = pd.read_parquet("data/train_series.parquet", columns=["id", "time", "value"])
S = S[S.id.isin(ids)].sort_values(["id", "time"])
V = {k: g.to_numpy() for k, g in S.groupby("id")["value"]}
out = dict(zip(V, Parallel(n_jobs=30)(delayed(em.one)(v) for v in V.values())))
rest = [k for k, (w, ok) in out.items() if not w.startswith("SIN") and not any(ok.values())]
good = [k for k, (w, ok) in out.items() if w == "AR1" and ok["AR1"]]
r = np.random.default_rng(3)
pick = [(k, "UNMATCHED") for k in r.choice(rest, 6, replace=False)] + [(k, "matched AR1") for k in r.choice(good, 2, replace=False)]
fig, ax = plt.subplots(len(pick), 3, figsize=(16, 2.1 * len(pick)), gridspec_kw={"width_ratios": [5, 1.4, 1.4]})
for i, (k, tag) in enumerate(pick):
    v = V[k]
    c = "tab:red" if tag == "UNMATCHED" else "tab:blue"
    ax[i, 0].plot(v, lw=0.6, color=c)
    ax[i, 0].set_title(f"{tag}  id={k}  n={len(v)}", fontsize=9, loc="left")
    ax[i, 1].bar(range(1, 41), acf(v), color=c, width=0.8)
    ax[i, 1].axhline(0, color="k", lw=0.5)
    ax[i, 1].set_title("autocorrelation, lag 1-40", fontsize=8)
    pw = np.abs(np.fft.rfft(v - v.mean())) ** 2
    ax[i, 2].plot(np.fft.rfftfreq(len(v))[1:], pw[1:], lw=0.6, color=c)
    ax[i, 2].set_title("spectrum (frequency 0-0.5)", fontsize=8)
fig.tight_layout()
fig.savefig("outputs/probe/unmatched_9pct.png", dpi=90)
print(len(rest), "unmatched,", len(good), "AR1-matched; saved outputs/probe/unmatched_9pct.png")
