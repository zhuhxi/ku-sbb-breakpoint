"""Derived from FigMirror C_curves (shipped iter 0): same style, generator-realism data (not a separate FigMirror run).
Style: Chronos (Ansari et al. 2024) model-size line chart. Data: KU-SBB submissions OOF vs public LB AUC.
Render: /home/zhuhx/miniconda3/bin/python3 figure_iter0.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text, Annotation
import matplotlib.patheffects as pe

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

# === DATA SECTOR (edit here) ===
X_LABELS = ["v1", "v2", "v3", "v4", "v5", "v6", "final"]
X_TITLE = "Generator version"
Y_TITLE = "Real-vs-fake AUC"
LEGEND_TITLE = "Realism (lower = more realistic)"
SERIES = [("Real vs synthetic, mean of 3 seeds", [0.6619, 0.6211, 0.6037, 0.5949, 0.5818, 0.5792, 0.5797])]
FINAL_LABELS = ["0.580"]
# === END DATA SECTOR ===

# ---------------- style constants (see notes_iter0.md for provenance) ----------------
DPI = 200
FIGSIZE = (7.2, 6.5)            # aspect 1.108; L1 ref aspect 2326/2108 = 1.103
# Colors
COL_OOF = "#008fd5"   # L1-PIL: blue series line, median 8x8 window at ref (858,1646)
COL_LB = "#fc4f30"    # L1-PIL: red/orange series, median over all r>230,g<110,b<80 pixels
COL_SPINE = "#000000" # L1 min-along-line: spine core pixels (0,0,0); L2 near-black class
COL_GRID = "#cccccc"  # L1-PIL: 6-px-wide solid gridline rows read (204,204,204)
COL_LEGEDGE = "#d6d6d6"  # L1-PIL: legend frame 8-px edge reads (214,214,214)
COL_TEXT = "#000000"  # L1: black bold text
# Widths (ref px scaled by k = 1440/2326 = 0.619, then px -> pt at 200 dpi)
LW_SPINE = 1.75       # ref 8 px
LW_GRID = 1.3         # ref 6 px
LW_LINE = 3.5         # ref 16 px
MS = 8.3              # ref marker diameter 38 px
LW_LEGFRAME = 1.7     # ref 8 px
TICK_LEN = 5.2        # ref 24 px major ticks, outward
TICK_W = 1.1          # ref 5 px
# Type: bold grotesque sans (ref looks Roboto Bold; not installed -> Liberation Sans Bold)
FS_TICK = 18.5        # ref tick cap height 60 px
FS_LEG = 18.0         # ref legend cap height 58 px (0.97 x tick)
FS_AXLABEL = 25.0     # ref axis-label cap ~80-85 px (~1.35 x tick)
FS_VALUE = 16.0       # final-value labels (not in ref; one notch under tick size)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
ax = fig.add_axes([0.205, 0.140, 0.755, 0.668])

xs = list(range(len(X_LABELS)))
colors = [COL_OOF]
handles = []
for (lab, ys), c in zip(SERIES, colors):
    h, = ax.plot(xs, ys, color=c, lw=LW_LINE, marker="o", ms=MS, mec=c, mfc=c,
                 solid_capstyle="round", zorder=3, label=lab, clip_on=False)
    handles.append(h)

# axes limits: x margin ~4.3% of width each side (ref 20M/710M inset 0.042/0.045)
ax.set_xlim(-0.33, len(xs) - 1 + 0.33)
ax.set_ylim(0.495, 0.675)
ax.set_xticks(xs)
ax.set_xticklabels(X_LABELS)
yt = [0.50, 0.55, 0.60, 0.65]
ax.set_yticks(yt)
ax.set_yticklabels([f"{v:.2f}" for v in yt]); ax.axhline(0.5, color="#808080", ls="--", lw=1.6, zorder=2); ax.text(0.15, 0.503, "0.5 = indistinguishable", color="#555555", fontsize=FS_VALUE, fontweight="bold", va="bottom", zorder=6)

# all four spines, black (L1: rectangle frame)
for s in ax.spines.values():
    s.set_visible(True)
    s.set_color(COL_SPINE)
    s.set_linewidth(LW_SPINE)
    s.set_zorder(4)

# ticks: outward major ticks on left and bottom only (L1)
ax.tick_params(axis="both", which="major", direction="out", length=TICK_LEN, width=TICK_W,
               color=COL_SPINE, labelsize=FS_TICK, pad=5, top=False, right=False)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontweight("bold")

# gridlines: solid light grey, both directions, below data
ax.grid(True, which="major", axis="both", color=COL_GRID, lw=LW_GRID, ls="-")
ax.set_axisbelow(True)

ax.set_xlabel(X_TITLE, fontsize=FS_AXLABEL, fontweight="bold", labelpad=8)
ax.set_ylabel(Y_TITLE, fontsize=FS_AXLABEL, fontweight="bold", labelpad=10)

# final-value labels: series-colored bold, left of the last marker (ha='right' keeps inside axes)
last = len(xs) - 1
offsets = [(12, 14), (12, 10)]   # OOF below, LB above; ha=right keeps label inside axes
vas = ["bottom", "bottom"]
for (lab, ys), c, txt, off, va in zip(SERIES, colors, FINAL_LABELS, offsets, vas):
    a = ax.annotate(txt, xy=(last, ys[-1]), xytext=off, textcoords="offset points",
                    ha="right", va=va, fontsize=FS_VALUE, fontweight="bold", color=c,
                    zorder=6)
    a.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

# legend: outside above axes, titled, rounded light-grey frame, spans most of the width
leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.561, 0.992),
                 ncol=1, title=LEGEND_TITLE, fontsize=FS_LEG, title_fontsize=FS_LEG,
                 frameon=True, fancybox=True, edgecolor=COL_LEGEDGE, facecolor="white",
                 framealpha=1.0, handlelength=2.1, handletextpad=0.7, borderpad=0.45,
                 labelspacing=0.35, borderaxespad=0.0)
leg.get_frame().set_linewidth(LW_LEGFRAME)
leg.get_title().set_fontweight("bold")
for t in leg.get_texts():
    t.set_fontweight("bold")
for t in ax.texts + ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]:
    t.set_zorder(max(t.get_zorder(), 6))

OUT = "figure.png"
fig.savefig(OUT, dpi=DPI, facecolor="white"); fig.savefig("figure.pdf", dpi=DPI, facecolor="white"); fig.savefig("output.png", dpi=DPI, facecolor="white")


# ---------------- floor self-check ----------------
def floor_check(fig, ax, leg):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lines = []
    texts = []
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        if t.get_text():
            texts.append(("tick", t))
    texts += [("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)]
    texts += [("annot", t) for t in ax.texts]
    texts += [("legend", t) for t in leg.get_texts()] + [("legtitle", leg.get_title())]
    bbs = [(k, t, t.get_window_extent(r)) for k, t in texts]
    ov = []
    for i, (ka, ta, ba) in enumerate(bbs):
        for kb, tb, bb in bbs[i + 1:]:
            if ba.overlaps(bb):
                ov.append(f"{ka}('{ta.get_text()}') x {kb}('{tb.get_text()}')")
    lines.append("text_overlap: " + ("PASS" if not ov else "FAIL " + "; ".join(ov)))
    fb = fig.bbox
    clip = [t.get_text() for _, t, b in bbs
            if b.x0 < fb.x0 or b.y0 < fb.y0 or b.x1 > fb.x1 or b.y1 > fb.y1]
    lb = leg.get_window_extent(r)
    if lb.x0 < 0 or lb.y0 < 0 or lb.x1 > fb.x1 or lb.y1 > fb.y1:
        clip.append("legend frame")
    lines.append("clipping: " + ("PASS" if not clip else "FAIL " + ", ".join(clip)))
    xl = ax.xaxis.label.get_window_extent(r)
    lines.append(f"xlabel_y0>=0: {'PASS' if xl.y0 >= 0 else 'FAIL'} (y0={xl.y0:.1f}px)")
    axb = ax.get_window_extent(r)
    lines.append("legend_vs_axes: " + ("PASS" if lb.y0 > axb.y1 else "FAIL legend overlaps axes")
                 + f" (gap={lb.y0 - axb.y1:.1f}px)")
    # annotations inside axes
    ann_out = [t.get_text() for t in ax.texts
               if not (axb.x0 <= t.get_window_extent(r).x0 and t.get_window_extent(r).x1 <= axb.x1
                       and axb.y0 <= t.get_window_extent(r).y0 and t.get_window_extent(r).y1 <= axb.y1)]
    lines.append("annot_inside_axes: " + ("PASS" if not ann_out else "FAIL " + ", ".join(ann_out)))
    # text vs marks: annotation bbox vs data line vertices (display) and markers
    import numpy as np
    hit = []
    for t in ax.texts:
        b = t.get_window_extent(r)
        for ln in ax.lines:
            xy = ln.get_transform().transform(ln.get_xydata())
            # densify segments
            pts = []
            for p, q in zip(xy[:-1], xy[1:]):
                for s in np.linspace(0, 1, 60):
                    pts.append(p + s * (q - p))
            pts = np.array(pts)
            half = ln.get_linewidth() * fig.dpi / 72 / 2 + ln.get_markersize() * fig.dpi / 72 / 2 * 0
            inside = ((pts[:, 0] > b.x0 - half) & (pts[:, 0] < b.x1 + half)
                      & (pts[:, 1] > b.y0 - half) & (pts[:, 1] < b.y1 + half))
            mk = xy
            mr = ln.get_markersize() * fig.dpi / 72 / 2
            minside = ((mk[:, 0] > b.x0 - mr) & (mk[:, 0] < b.x1 + mr)
                       & (mk[:, 1] > b.y0 - mr) & (mk[:, 1] < b.y1 + mr))
            if inside.any() or minside.any():
                hit.append(f"'{t.get_text()}' x line {ln.get_label()[:12]}")
    maxz = max(a.get_zorder() for a in ax.get_children() if not isinstance(a, Text))
    low = [t.get_text() for t in ax.texts + ax.get_xticklabels() + ax.get_yticklabels()
           if t.get_text() and t.get_zorder() <= maxz]
    status = "PASS" if not hit and not low else "FAIL " + "; ".join(hit + [f"low z: {x}" for x in low])
    lines.append("text_obscured_by_marks: " + status)
    lines.append(f"figure_aspect: {FIGSIZE[0] / FIGSIZE[1]:.3f} (ref 1.103, band +-10%)")
    lines.append(f"panel_aspect: {axb.width / axb.height:.3f} (ref ~1.34 from spine bbox)")
    return lines


res = floor_check(fig, ax, leg)
with open("floor_selfcheck_final.txt", "w") as f:
    f.write("\n".join(res) + "\n")
print("\n".join(res))
