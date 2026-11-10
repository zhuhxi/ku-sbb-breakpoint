"""Slide variant of the shipped figure.py: same style and data, smaller canvas so text reads larger on a 16:9 slide."""
"""FigMirror F_limits, iter 1: Chronos-style ranked horizontal bars, our AUC data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text

plt.rcParams["pdf.fonttype"] = 42  # camera-ready: no Type 3
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

# === DATA SECTOR (edit here) ===
PANELS = [
    {
        "ylabel": "Series type",        # panel name 'By series type' -> L1 slot = per-panel ylabel ("Model")
        "xlabel": "AUC",
        "items": [("Sine", 0.911), ("AR(2)", 0.885), ("White noise", 0.874),
                  ("AR(1)", 0.869), ("Random walk", 0.709)],
        "hard": {"Random walk"},
    },
    {
        "ylabel": "Series length",
        "xlabel": "AUC",
        "items": [("Longest 20%", 0.911), ("4th quintile", 0.894), ("3rd quintile", 0.880),
                  ("2nd quintile", 0.847), ("Shortest 20%", 0.821)],
        "hard": {"Shortest 20%"},
    },
]
XLIM = (0.5, 1.0)
XTICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LEGEND = [("Hard cases", "hard"), ("Others", "other")]
# === END DATA SECTOR ===

# --- palette ---
COL_HARD = "#e9816e"   # L1-PIL: salmon bar fill, median 8x8 window at (540,390) px in reference_clean
COL_OTHER = "#499ec8"  # L1-PIL: blue bar fill, median 8x8 window at (600,918) px in reference_clean
COL_SPINE = "#000000"  # L1 min-along-line: spine core pixels = 0 (near-black hairline class)
COL_VALUE = "#444444"  # L1-PIL: median of dark pixels inside value labels = (68,68,68)
COL_TICKTXT = "#333333"  # L1-PIL: median dark pixels of regular-model y labels = (51,51,51)
COL_LABEL = "#000000"  # L1-PIL: axis labels / emphasised y labels are pure black
COL_LEGEDGE = "#d6d6d6"  # L1 min-along-line on legend frame: darkest 214 -> #d6d6d6
COLORS = {"hard": COL_HARD, "other": COL_OTHER}

# --- type: sans bold (reference is Roboto-class bold sans); Liberation Sans = available sans ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Roboto", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["font.weight"] = "bold"
TICK_PT = 10.5     # x tick numbers; L1: ref digit height 21 px at 200 dpi -> ~10.5 pt
YTICK_PT = 11.0    # category labels; L1: ref paren-height 28 px -> ~11 pt
VALUE_PT = 10.5   # value labels; L1: ref digit height 20 px -> ~10-10.5 pt
LEGEND_PT = 11.0  # L1: ref legend cap 22 px -> ~11 pt
AXLABEL_PT = 15.0  # L1: ref xlabel 38 px w/ descender -> ~15 pt; ratio to tick 1.43 kept

DPI = 200
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), dpi=DPI)
fig.subplots_adjust(left=0.17, right=0.985, bottom=0.16, top=0.86, wspace=0.50)  # iter1: re-tightened after type shrink

for ax, p in zip(axes, PANELS):
    names = [n for n, _ in p["items"]]
    vals = [v for _, v in p["items"]]
    ys = list(range(len(names)))[::-1]  # best on top
    cols = [COL_HARD if n in p["hard"] else COL_OTHER for n in names]
    ax.barh(ys, [v - XLIM[0] for v in vals], left=XLIM[0], height=0.8,
            color=cols, edgecolor="none", zorder=2)
    for y, v in zip(ys, vals):
        ax.annotate(f"{v:.3f}", xy=(v, y), xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=VALUE_PT, fontweight="bold",
                    color=COL_VALUE, zorder=5)
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.55, len(names) - 0.45)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{t:.1f}" for t in XTICKS])
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    for lab, n in zip(ax.get_yticklabels(), names):
        lab.set_color(COL_LABEL if n in p["hard"] else COL_TICKTXT)
    ax.tick_params(axis="x", labelsize=TICK_PT, length=4, width=1.0, direction="out",
                   pad=3, colors=COL_SPINE, labelcolor=COL_LABEL)
    ax.tick_params(axis="y", labelsize=YTICK_PT, length=4, width=1.0, direction="out",
                   pad=3, colors=COL_SPINE)
    for s in ax.spines.values():  # L1: all four spines, near-black
        s.set_visible(True)
        s.set_color(COL_SPINE)
        s.set_linewidth(1.1)
        s.set_zorder(3)
    ax.grid(False)  # L1: no gridlines in reference
    ax.set_xlabel(p["xlabel"], fontsize=AXLABEL_PT, fontweight="bold", color=COL_LABEL, labelpad=4)
    ax.set_ylabel(p["ylabel"], fontsize=AXLABEL_PT, fontweight="bold", color=COL_LABEL, labelpad=6)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_zorder(6)
    ax.xaxis.label.set_zorder(6)
    ax.yaxis.label.set_zorder(6)

handles = [Patch(facecolor=COLORS[k], edgecolor="none", label=l) for l, k in LEGEND]
leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2,
                 frameon=True, fancybox=True, edgecolor=COL_LEGEDGE, facecolor="white",
                 framealpha=1.0, prop={"size": LEGEND_PT, "weight": "bold"},
                 handlelength=2.6, handleheight=0.9, handletextpad=0.8, columnspacing=2.2,
                 borderpad=0.35)
leg.get_frame().set_linewidth(1.2)


# ---------------- floor self-check ----------------
def floor_checks(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lines = []
    texts = []
    for ax in fig.axes:
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text():
                texts.append(("tick", t))
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if t.get_text():
                texts.append(("label", t))
        for c in ax.get_children():
            if isinstance(c, matplotlib.text.Annotation) and c.get_text():
                texts.append(("annot", c))
    for t in leg.get_texts():
        texts.append(("legend", t))
    bbs = [(k, t, t.get_window_extent(r)) for k, t in texts]
    ov = []
    for i, (ka, ta, ba) in enumerate(bbs):
        for kb, tb, bb in bbs[i + 1:]:
            if ba.overlaps(bb):
                ov.append(f"{ka}('{ta.get_text()}') x {kb}('{tb.get_text()}')")
    lines.append("text_overlap: " + ("PASS" if not ov else "FAIL " + "; ".join(ov)))
    clip = [t.get_text() for _, t, b in bbs
            if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1]
    lb = leg.get_window_extent(r)
    if lb.x0 < 0 or lb.y1 > fig.bbox.y1:
        clip.append("legend")
    lines.append("clipped_text: " + ("PASS" if not clip else "FAIL " + ", ".join(clip)))
    # annotations must stay inside their own axes (no cross-panel bleed)
    bleed = []
    for ax in fig.axes:
        for c in ax.get_children():
            if isinstance(c, matplotlib.text.Annotation):
                b = c.get_window_extent(r)
                if b.x1 > ax.bbox.x1 - 1 or b.x0 < ax.bbox.x0:
                    bleed.append(c.get_text())
    lines.append("annot_inside_axes: " + ("PASS" if not bleed else "FAIL " + ", ".join(bleed)))
    # legend vs axes overlap
    lo = [i for i, ax in enumerate(fig.axes) if lb.overlaps(ax.bbox)]
    lines.append("legend_clear_of_axes: " + ("PASS" if not lo else f"FAIL axes {lo}"))
    # cross-panel: ax0 annotations vs ax1 y tick labels / ylabel
    a1 = fig.axes[1]
    left_items = [c.get_window_extent(r) for c in fig.axes[0].get_children()
                  if isinstance(c, matplotlib.text.Annotation)]
    right_items = [t.get_window_extent(r) for t in a1.get_yticklabels() if t.get_text()] + \
                  [a1.yaxis.label.get_window_extent(r)]
    cp = any(lb_.x1 >= rb.x0 for lb_ in left_items for rb in right_items
             if lb_.y0 < rb.y1 and lb_.y1 > rb.y0)
    lines.append("cross_panel: " + ("FAIL" if cp else "PASS"))
    xl_ok = all(ax.xaxis.label.get_window_extent(r).y0 >= 0 for ax in fig.axes)
    lines.append("xlabel_y0_ge_0: " + ("PASS" if xl_ok else "FAIL"))
    # layering: text z above every non-text artist in axes
    bad = []
    for ax in fig.axes:
        mz = max([a.get_zorder() for a in ax.get_children() if not isinstance(a, Text)] or [0])
        for t in list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels() + \
                [ax.xaxis.label, ax.yaxis.label]:
            if t.get_text() and t.get_zorder() <= mz:
                bad.append(t.get_text())
    # value labels sit in white space right of bars: verify no bar patch under label bbox
    for ax in fig.axes:
        anns = [c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)]
        for c in anns:
            b = c.get_window_extent(r)
            for patch in ax.patches:
                pb = patch.get_window_extent(r)
                if b.overlaps(pb):
                    bad.append(c.get_text() + "(on bar)")
    lines.append("text_obscured_by_marks: " + ("PASS" if not bad else "FAIL " + ", ".join(bad)))
    W, H = fig.get_size_inches()
    lines.append(f"figure_aspect: {W / H:.3f} (ref 2.277, band 2.05-2.50) "
                 + ("PASS" if 2.05 <= W / H <= 2.50 else "FAIL"))
    return lines


if __name__ == "__main__":
    res = floor_checks(fig)
    fig.savefig("figure_slide.png", dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    with open("floor_selfcheck_slide.txt", "w") as fh:
        fh.write("Floor self-check, iter 1 (figure_iter1.py, rendered bboxes after canvas.draw)\n")
        fh.write("\n".join(res) + "\n")
        fh.write("overall: " + ("PASS" if all("FAIL" not in l for l in res) else "FAIL") + "\n")
    print("\n".join(res))
