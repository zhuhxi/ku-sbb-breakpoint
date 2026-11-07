"""In which unit is the level jump set? For every real break series: the jump (median after - median before)
divided by (a) the noise sd (AR residual sd before the boundary) and (b) the series sd before the boundary, plotted
against the lag-1 autocorrelation before the boundary. A unit the generator uses shows a flat band (e.g. at 2) that
does not depend on the autocorrelation; the other unit bends with it. Menu "L abc40 ns35" (noise unit) as reference.

    python probe/level_unit_plot.py
"""
import sys
import time

sys.argv = [sys.argv[0], "--menu", "L abc40 ns35", "--only-break", "1"]
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

s = importlib.util.spec_from_file_location("bc", "probe/break_c2st.py")
bc = importlib.util.module_from_spec(s); s.loader.exec_module(bc)


def units(v, n0):
    a, b = v[:n0], v[n0:]
    X = np.column_stack([np.ones(n0 - 2), a[1:-1], a[:-2]])
    coef, *_ = np.linalg.lstsq(X, a[2:], rcond=None)
    sn = (a[2:] - X @ coef).std()
    c = a - a.mean()
    r1 = (c[1:] @ c[:-1]) / (c @ c)
    d = abs(np.median(b) - np.median(a))
    return dict(r1=r1, noise=d / sn, series=d / a.std(), ratio=a.std() / sn, cls=bc.cb.bs.one(v, n0)["cls"])


t0 = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
S = S[S.id.isin(set(lab.index[lab == 1]))].sort_values(["id", "time"])
real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S.groupby("id")]
fake = Parallel(n_jobs=30)(delayed(bc.cb.gen)(bc.MENU, 8 * 10 ** 6 + i, True, True) for i in range(len(real)))
D = {k: pd.DataFrame(Parallel(n_jobs=30, batch_size=256)(delayed(units)(v, n0) for v, n0 in L)) for k, L in [("real", real), ("fake (noise unit)", fake)]}
fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
for i, (k, d) in enumerate(D.items()):
    for j, u in enumerate(["noise", "series"]):
        a = ax[i, j]
        a.scatter(d.r1, d[u].clip(0, 4), s=2, alpha=.25)
        a.axhline(2, color="r", lw=.8)
        a.set_ylim(0, 4)
        a.set_title(f"{k}: jump / {u} sd", fontsize=10)
        a.set_xlabel("lag-1 autocorrelation before the boundary")
        big = d[d[u] > 1.3]                                    # the large group: median jump by autocorrelation band
        med = big.groupby(pd.cut(big.r1, [-1, -.4, -.1, .1, .4, .7, 1]), observed=True)[u].median()
        print(f"{k:18s} jump/{u:6s} sd, median of jumps > 1.3 by r1 band: " + "  ".join(f"{iv.left:+.1f}..{iv.right:+.1f}: {m:.2f}" for iv, m in med.items()))
D = {k: d[d.cls.isin(["WN", "AR1", "AR2"])] for k, d in D.items()}   # same structures on both sides
for k, d in D.items():                                        # share of large jumps (> 1.5 noise sd) by autocorrelation band
    g = d.groupby(pd.cut(d.r1, [-1, -.25, -.1, .1, .25, .75]), observed=True)["noise"].apply(lambda z: (z.between(1.5, 3)).mean() * 100)
    print(f"{k:18s} % with a large jump (1.5-3 noise sd): " + "  ".join(f"{iv.left:+.2f}..{iv.right:+.2f}: {m:.1f}" for iv, m in g.items()))
fig.tight_layout()
fig.savefig("outputs/probe/level_unit.png", dpi=80)
print(f"real {len(real)} series, total {time.time() - t0:.0f}s")
