"""Picture: why a large share of break series cannot be told apart. Left: a big and a small level break (simulated).
Right: real data, strongest of the three changes (level / spread / dependence) in units of the no-break spread.

    python probe/plot_hidden.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

r = np.random.default_rng(5)
n, n0 = 500, 250
e = r.standard_normal(n)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.2]})
for a, d, title in [(ax[0], 2.0, "BIG break: level moves 2.0 sd  -> easy to see"),
                    (ax[1], 0.2, "SMALL break: level moves 0.2 sd  -> hidden in noise")]:
    x = e + d * (np.arange(n) >= n0)
    a.plot(x, lw=0.6, color="0.3")
    a.hlines([0, d], [0, n0], [n0, n], colors="tab:red", lw=2.5)
    a.axvline(n0, color="tab:blue", ls="--")
    a.set_title(title, fontsize=11)
    a.set_ylim(-4, 5.5)
D = pd.read_parquet("outputs/probe/break_size.parquet")
Z = pd.DataFrame({c: D[c].abs() / D.loc[D.y == 0, c].abs().quantile(0.5) for c in ["level", "spread", "dep"]})
s = Z.max(axis=1)
b = np.linspace(0, 12, 49)
ax[2].hist(s[D.y == 0], b, alpha=0.6, density=True, label="no break (label 0)", color="tab:blue")
ax[2].hist(s[D.y == 1], b, alpha=0.6, density=True, label="break (label 1)", color="tab:red")
h0, _ = np.histogram(s[D.y == 0].clip(upper=11.99), b, density=True)
h1, _ = np.histogram(s[D.y == 1].clip(upper=11.99), b, density=True)
hidden = np.minimum(h0, h1).sum() * (b[1] - b[0])           # overlap area: breaks that look like no-break
ax[2].fill_between(b[:-1], 0, np.minimum(h0, h1), step="post", color="none", hatch="///", edgecolor="k", lw=0)
ax[2].text(5, 0.33, f"hatched = overlap = {hidden:.0%} of breaks\nlook exactly like no-break series", fontsize=10)
ax[2].set_title("REAL data: biggest of the 3 changes per series", fontsize=11)
ax[2].set_xlabel("size of change (1 = typical no-break wobble)")
ax[2].legend()
fig.tight_layout()
fig.savefig("outputs/probe/hidden_breaks.png", dpi=90)
print(f"hidden share {hidden:.3f}")
