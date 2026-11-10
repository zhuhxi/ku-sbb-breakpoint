"""Cover and closing visuals in the deck's Chronos-derived palette (same colours/type as figmirror/G_pipeline).
python reports/final/hero/make_hero.py  -> hero/cover.png, hero/loop.png"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]})
BLUE, RED, PINK, PURPLE, GREY, EDGE, TEXT, SUB, TOK = "#0e87cc", "#fd5956", "#fbe6f0", "#874289", "#f5f5f5", "#5e5e5e", "#111111", "#5e5e5e", "#e5f1fb"
OUT = "reports/final/hero/"
S = pd.read_parquet("outputs/aug/series_simr10a.parquet"); M = pd.read_parquet("outputs/aug/meta_simr10a.parquet")
def ser(i, step=2):
    x = S[S.id == i].sort_values("time"); v = x.value.to_numpy(); n0 = int((x.period == 0).sum()); return v[::step], n0 // step
def box(ax, x, y, w, h, fc, ec=EDGE, lw=2.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=lw, ls=ls, zorder=1))
def glyph(ax, v, n0, x, y, w, h, lw=1.4, dash=True):
    t = np.linspace(x, x + w, len(v)); z = (v - v.mean()) / (np.ptp(v) + 1e-9) * h + y + h / 2
    ax.plot(t[:n0 + 1], z[:n0 + 1], color=BLUE, lw=lw, zorder=3); ax.plot(t[n0:], z[n0:], color=RED, lw=lw, zorder=3)
    if dash: ax.plot([t[n0]] * 2, [y - .05, y + h + .05], color="#808080", ls="--", lw=1.3, zorder=4)
def arrow(ax, x0, x1, y, lab=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=28, lw=3, color=EDGE, zorder=2))
    if lab: ax.text((x0 + x1) / 2, y + .22, lab, ha="center", va="bottom", fontsize=13, color=PURPLE)

# ---------------- cover: the one idea, left to right ----------------
fig = plt.figure(figsize=(12.4, 3.6), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12.4); ax.set_ylim(0, 3.6); ax.axis("off")
# 1 real data
box(ax, 0.1, 0.55, 2.5, 2.3, TOK); v, n0 = ser("simr10a000871", 3); glyph(ax, v, n0, 0.3, 1.25, 2.1, 1.1)
ax.text(1.35, 3.05, "Real data", ha="center", fontsize=17, weight="bold", color=SUB); ax.text(1.35, 0.8, "24,000 labelled series", ha="center", fontsize=13, color=TEXT)
arrow(ax, 2.7, 3.55, 1.7, "fit")
# 2 generator (accent)
box(ax, 3.6, 0.95, 2.4, 1.5, PINK); ax.text(4.8, 1.88, "Reverse-engineered", ha="center", fontsize=15, weight="bold", color=PURPLE); ax.text(4.8, 1.48, "generator", ha="center", fontsize=15, weight="bold", color=PURPLE)
ax.text(4.8, 3.05, "Generator", ha="center", fontsize=17, weight="bold", color=SUB); ax.text(4.8, 0.62, "real-vs-fake AUC 0.58", ha="center", fontsize=13, color=TEXT)
arrow(ax, 6.1, 6.95, 1.7, "sample")
# 3 deck of synthetic series with known breaks
ids = M[(M.label == 1) & M.kind.isin(["level", "spread", "dep", "shape"])].index[[5, 40, 80, 120]]
for k, i in enumerate(ids[::-1]):
    dx, dy = 0.16 * (3 - k), 0.16 * (3 - k)
    box(ax, 7.05 + dx, 0.95 + dy, 2.0, 1.25, "white", lw=1.4)
    if k == 3: v, n0 = ser(i, 3); glyph(ax, v, n0, 7.15, 1.1, 1.8, 0.95, lw=1.0, dash=True)
ax.text(8.3, 3.05, "Synthetic data", ha="center", fontsize=17, weight="bold", color=SUB); ax.text(8.3, 0.62, "unlimited, every break known", ha="center", fontsize=13, color=TEXT)
arrow(ax, 9.65, 10.4, 1.7, "teach")
# 4 model + result
box(ax, 10.5, 0.95, 1.8, 1.5, GREY); ax.text(11.4, 1.88, "Detectors +", ha="center", fontsize=14, weight="bold", color=TEXT); ax.text(11.4, 1.5, "TabPFN", ha="center", fontsize=14, weight="bold", color=TEXT)
ax.text(11.4, 3.05, "Model", ha="center", fontsize=17, weight="bold", color=SUB); ax.text(11.4, 0.62, "public AUC 0.8779", ha="center", fontsize=13, color=RED, weight="bold")
fig.savefig(OUT + "cover.png", facecolor="white"); plt.close(fig)

# ---------------- closing: the loop that produced every gain ----------------
fig = plt.figure(figsize=(12.4, 4.6), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12.4); ax.set_ylim(0, 4.6); ax.axis("off")
W, H, Y = 3.3, 2.9, 1.15; X = [0.35, 4.55, 8.75]
heads = [("1  Rebuild the data", "real-vs-fake AUC 0.66 → 0.58"), ("2  Learn without labels", "zero-shot detector AUC 0.85"), ("3  Measure what is fixable", "oracle: noise shape worth +0.007")]
for k, (x, (h, cap)) in enumerate(zip(X, heads)):
    box(ax, x, Y, W, H, PINK if k == 1 else (TOK if k == 0 else GREY))
    ax.text(x + W / 2, Y + H - 0.38, h, ha="center", fontsize=16, weight="bold", color=PURPLE if k == 1 else TEXT)
    ax.text(x + W / 2, Y + 0.25, cap, ha="center", fontsize=13, color=TEXT)
# 1: realism curve
r = [0.6619, 0.6211, 0.6037, 0.5949, 0.5818, 0.5792, 0.5797]; xs = np.linspace(X[0] + .45, X[0] + W - .45, len(r)); ys = Y + 0.75 + (np.array(r) - .55) / .12 * 1.25
ax.plot([xs[0] - .1, xs[-1] + .1], [Y + .75 + (.5 - .55) / .12 * 1.25 + .3] * 2, color="#808080", ls="--", lw=1.2)
ax.plot(xs, ys, color=BLUE, lw=3, marker="o", ms=6, zorder=3); ax.text(xs[-1] + .12, Y + .75 + (.5 - .55) / .12 * 1.25 + .38, "0.5", fontsize=11, color="#808080")
# 2: synthetic series -> detector score
v, n0 = ser("simr10a000067", 2); glyph(ax, v, n0, X[1] + .35, Y + 1.0, 1.55, 0.95, lw=1.0)
arrow(ax, X[1] + 2.0, X[1] + 2.45, Y + 1.47); ax.text(X[1] + 2.85, Y + 1.47, "P(break)", ha="center", va="center", fontsize=12, weight="bold", color=RED)
# 3: oracle bars
g = [0.0070, 0.0023, 0.0016, 0.0013, 0.0011]; by = np.linspace(Y + 2.05, Y + 0.75, len(g))
for j, (gg, yy) in enumerate(zip(g, by)):
    ax.add_patch(plt.Rectangle((X[2] + .45, yy - .1), gg / .0070 * 2.3, .2, fc="#e9816e" if j == 0 else "#499ec8", ec="none", zorder=3))
# arrows between steps and the return loop
arrow(ax, X[0] + W + .08, X[1] - .08, Y + H / 2); arrow(ax, X[1] + W + .08, X[2] - .08, Y + H / 2)
ax.add_patch(FancyArrowPatch((X[2] + W / 2, Y - .05), (X[0] + W / 2, Y - .05), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>", mutation_scale=28, lw=2.5, color=PURPLE, zorder=2))
ax.text(6.2, 0.12, "the measured gap tells us what to fix next", ha="center", fontsize=14, color=PURPLE, style="italic")
fig.savefig(OUT + "loop.png", facecolor="white"); plt.close(fig); print("ok")
