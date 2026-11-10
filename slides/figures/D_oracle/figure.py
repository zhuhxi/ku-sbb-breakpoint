"""FigMirror D_oracle, iter 2: oracle AUC-gain horizontal bars in the Chronos
(Ansari et al. 2024) horizontal-bar style. Self-contained."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

# === DATA SECTOR (edit here) ===
# (label shown on y axis, AUC gain); sorted largest first, drawn top-down
DATA = [
    ("Noise type + tail\n(df)",          0.0070),
    ("Volatility clustering\n(GARCH)",   0.0023),
    ("Trend",                            0.0016),
    ("Series type",                      0.0013),
    ("AR coefficients",                  0.0011),
    ("Sine parameters",                  0.0011),
]
HIGHLIGHT_INDEX = 0                      # top bar (noise type) gets the accent colour
XLABEL = "AUC gain if known (base 0.8154)"
YLABEL = "Hidden parameter"
VALUE_FMT = "+{:.4f}"
LEGEND_LABELS = ("Largest gain", "Other parameters")  # key for accent vs muted
# === END DATA SECTOR ===

# --- palette / structure (see notes_iter0.md) ---
COL_ACCENT = "#e9816e"  # L1-PIL: salmon series, median 10x10 at ref (936,439) px
COL_MUTED  = "#499ec8"  # L1-PIL: blue series, median 10x10 at ref (936,357) px
COL_VALUE  = "#444444"  # L1-PIL: darkest-ink median of value label '0.603'
COL_INK    = "#000000"  # L1-PIL: darkest-ink median of tick / category labels
COL_LEGEND_EDGE = "#d6d6d6"  # L1-PIL: legend frame edge, darkest-ink along frame
COL_SPINE  = "#000000"  # L2-class near-black (#000-#444); L1 frame reads pure black
SPINE_LW   = 1.6        # L1-perceived heavy frame (~7 px of 1767 px ref); above L2 hairline range
TICK_LEN   = 6.0        # L1-perceived outward ticks (~28 px ref)

# --- type: reference is a bold grotesk (Roboto Bold); not installed -> Liberation Sans Bold ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Roboto", "Liberation Sans", "Arial", "Nimbus Sans", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
TICK_PT  = 21           # iter2: reviewer L1 box 1-2, body text scaled up toward ref loud register
VALUE_PT = 21           # L1: value labels same height as ticks (53 vs 53 px)
LABEL_PT = 29           # title:tick 1.38 (reviewer: ref tick ~0.7 of title)

DPI = 180
FIGSIZE = (9.8, 7.2)    # aspect 1.36, within +-10% of L1 1.324

labels = [d[0] for d in DATA]
vals = [d[1] for d in DATA]
n = len(DATA)
ypos = list(range(n))[::-1]            # first item at top
BAR_H = 0.80                           # L1: bar group fills 242 of 303 px slot (0.80) at ref x=700

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.subplots_adjust(left=0.355, right=0.97, top=0.88, bottom=0.155)

colors = [COL_ACCENT if i == HIGHLIGHT_INDEX else COL_MUTED for i in range(n)]
bars = ax.barh(ypos, vals, height=BAR_H, color=colors, edgecolor="none", zorder=2)
# framed legend above the axes (L1: rounded light-grey frame, edge ~#d6d6d6, bold entries)
from matplotlib.patches import Patch
leg = ax.legend([Patch(color=COL_ACCENT), Patch(color=COL_MUTED)], LEGEND_LABELS,
                loc="lower right", bbox_to_anchor=(1.0, 1.03), ncol=2, frameon=True,
                fancybox=True, edgecolor=COL_LEGEND_EDGE, framealpha=1.0,
                prop={"size": TICK_PT, "weight": "bold"}, handlelength=2.4, handleheight=1.0,
                handletextpad=0.6, columnspacing=1.8, borderpad=0.55, borderaxespad=0.0)
leg.get_frame().set_linewidth(1.2)
leg.set_zorder(7)

XMAX = 0.0090
ax.set_xlim(0, XMAX)
ax.set_ylim(-0.52, n - 0.48)   # L1: ~0.1 slot margin between outer bars and frame
ax.xaxis.set_major_locator(FixedLocator([0, 0.002, 0.004, 0.006, 0.008]))
ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
ax.set_yticks(ypos)
ax.set_yticklabels(labels, multialignment="right")

for y, v in zip(ypos, vals):
    ax.annotate(VALUE_FMT.format(v), xy=(v, y), xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=VALUE_PT, fontweight="bold",
                color=COL_VALUE, zorder=5)

# all four spines, heavy black frame (L1: box closed on all sides)
for side in ("left", "right", "top", "bottom"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color(COL_SPINE)
    ax.spines[side].set_linewidth(SPINE_LW)
ax.tick_params(axis="both", direction="out", length=TICK_LEN, width=SPINE_LW,
               color=COL_SPINE, labelsize=TICK_PT, labelcolor=COL_INK, pad=5)
ax.tick_params(top=False, right=False)
ax.grid(False)          # L1-PIL: 0 dark rows / 0 dark cols in panel interior
ax.set_axisbelow(True)

ax.set_xlabel(XLABEL, fontsize=LABEL_PT, color=COL_INK, labelpad=8)
ax.set_ylabel(YLABEL, fontsize=LABEL_PT, color=COL_INK, labelpad=6)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontweight("bold"); t.set_zorder(6)
ax.xaxis.label.set_zorder(6); ax.yaxis.label.set_zorder(6)

fig.savefig("figure.png", dpi=DPI, facecolor="white"); fig.savefig("figure.pdf", dpi=DPI, facecolor="white"); fig.savefig("output.png", dpi=DPI, facecolor="white")

# ---------------- floor self-check ----------------
def floor_check(fig, ax):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [("tick", t) for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text()]
    texts += [("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)]
    texts += [("annot", c) for c in ax.texts if c.get_text()]
    texts += [("legend", t) for t in leg.get_texts()]
    bbs = [(k, t, t.get_window_extent(r)) for k, t in texts]
    lines = []
    ov = []
    for i, (ka, ta, ba) in enumerate(bbs):
        for kb, tb, bb in bbs[i + 1:]:
            if ba.overlaps(bb):
                ov.append(f"{ka}({ta.get_text()!r}) x {kb}({tb.get_text()!r})")
    lines.append("text_overlap: " + ("PASS" if not ov else "FAIL " + "; ".join(ov)))
    fb = fig.bbox
    clip = [t.get_text() for _, t, b in bbs if b.x0 < fb.x0 or b.y0 < fb.y0 or b.x1 > fb.x1 or b.y1 > fb.y1]
    lines.append("clipped_text: " + ("PASS" if not clip else "FAIL " + repr(clip)))
    axb = ax.get_window_extent(r)
    out = [t.get_text() for k, t, b in bbs if k == "annot" and (b.x1 > axb.x1 - 2 or b.y0 < axb.y0 or b.y1 > axb.y1)]
    lines.append("annot_inside_axes: " + ("PASS" if not out else "FAIL " + repr(out)))
    # annotation must not sit on any bar
    hit = []
    for p in ax.patches:
        pb = p.get_window_extent(r)
        for k, t, b in bbs:
            if k == "annot" and b.overlaps(pb):
                hit.append(t.get_text())
    lines.append("text_obscured_by_marks: " + ("PASS" if not hit else "FAIL " + repr(hit)))
    maxz = max(p.get_zorder() for p in ax.patches)
    lowz = [t.get_text() for _, t, _ in bbs if t.get_zorder() <= maxz]
    lines.append("text_zorder_above_marks: " + ("PASS" if not lowz else "FAIL " + repr(lowz)))
    xl = ax.xaxis.label.get_window_extent(r)
    lines.append(f"xlabel_margins_px: left {xl.x0:.1f} right {fig.bbox.x1 - xl.x1:.1f} bottom {xl.y0:.1f}")
    lines.append(f"xlabel_y0_px: {ax.xaxis.label.get_window_extent(r).y0:.1f} (>=0 required)")
    lines.append(f"canvas_px: {fig.bbox.width:.0f}x{fig.bbox.height:.0f} aspect {fig.bbox.width/fig.bbox.height:.3f} (ref 1.324)")
    gap = min(axb.x1 - b.x1 for k, _, b in bbs if k == "annot")
    lines.append(f"min_annot_to_right_spine_px: {gap:.1f}")
    lines.append(f"axes_bbox_px: x0={axb.x0:.0f} x1={axb.x1:.0f} y0={axb.y0:.0f} y1={axb.y1:.0f}")
    lb = leg.get_window_extent(r)
    lines.append("legend_inside_canvas: " + ("PASS" if (lb.x0 >= 0 and lb.y1 <= fig.bbox.y1 and lb.x1 <= fig.bbox.x1) else "FAIL"))
    lines.append("legend_clear_of_axes: " + ("PASS" if lb.y0 > axb.y1 else "FAIL"))
    bh = [p.get_window_extent(r).height for p in ax.patches if hasattr(p, "get_height") and p.get_window_extent(r).y0 >= axb.y0]
    slot = axb.height / (n - 0.48 + 0.52)
    lines.append(f"bar_slot_fill: {bh[0]/slot:.2f} (ref 242/303 = 0.80)")
    lines.append(f"axes_aspect_w_over_h: {axb.width/axb.height:.3f} (ref ~1623/1208 = 1.34)")
    ok = all(("FAIL" not in l) for l in lines)
    lines.append("OVERALL: " + ("PASS" if ok else "FAIL"))
    return lines

if __name__ == "__main__":
    res = floor_check(fig, ax)
    with open("floor_selfcheck_final.txt", "w") as f:
        f.write("\n".join(res) + "\n")
    print("\n".join(res))
