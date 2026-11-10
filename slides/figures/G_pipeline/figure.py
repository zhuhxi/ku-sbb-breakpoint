"""FigMirror iter 1 -- KU-SBB pipeline schematic in the style of Chronos Fig. 1.

Render:  /home/zhuhx/miniconda3/bin/python3 figure_iter1.py  -> img_iter1.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib import font_manager
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

# === DATA SECTOR (edit here) ===
SERIES = [float(v) for v in (
    "-1.518,0.459,0.765,-0.850,-0.086,2.352,1.123,-0.051,0.345,0.190,-1.612,0.091,0.864,0.089,1.343,"
    "-0.857,0.792,0.572,0.510,1.563,1.385,1.716,-0.012,1.143,1.841,-0.472,1.195,-1.958,1.204,-1.050,"
    "0.231,2.041,-1.007,1.392,0.475,-0.031,0.167,1.189,-0.197,-0.186,1.242,-0.923,1.233,0.280,-1.090,"
    "-0.545,0.346,-1.273,0.864,-2.674,0.363,-0.060,-0.050,-1.063,-0.509,2.116,2.302,1.135,-0.500,-0.163,"
    "-0.397,0.729,0.017,0.124,0.314,-0.331,0.462,0.806,-0.242,-1.560,0.495,0.833,0.141,0.227,0.424,"
    "0.812,-1.190,0.036,-1.359,-1.536,0.399,-1.742,0.067,-1.159,-0.232,-0.681,0.735,-1.223,-0.457,-0.703,"
    "-0.817,2.166,-1.383,-0.618,-0.888,-1.016,-0.838,-0.278,1.321,0.228,-0.285,-1.576,-0.770,-2.172,-2.191,"
    "-1.329,-0.992,-0.084,-0.322,-1.208,1.054,-1.355,-0.029,-1.993,-1.375,-2.657,-0.893,-0.461,-0.581,-0.482,"
    "-0.310,-0.241,1.034,0.752,0.103,-0.019,-1.090,-1.300,-1.586,0.656,-1.727,-0.320,-1.457,-0.058,-0.477,"
    "-1.331,-0.382,0.864,-1.812,0.496,-0.772,-1.442,0.858,-0.285,0.514,-0.030,-0.039,-1.342,-2.217,0.448,"
    "-0.939,0.950,1.369,-0.043,-0.300,-0.774,-1.000,1.239,-0.900,-1.383,-1.476,0.332,0.121,-0.045,-1.313,"
    "-0.859,-1.895,-0.953,0.155,-0.978,-0.054,0.462,-0.117,-0.003,-0.152,0.292,-1.055,-0.246,-0.598,-0.085,"
    "-0.394,1.443,0.440,-1.495,-1.714,-2.223,1.201,-1.942,-1.964,1.380,-0.979,-0.832,1.172,-0.226,0.422,"
    "-1.207,0.132,-2.941,-1.242,-1.010,0.266,0.685,1.202,-0.772").split(",")]
N0 = 76  # boundary index in the thinned series

COL_TITLES = ["Real series → features",
              "Generator → label-free detectors",
              "Ensemble → prediction"]
# column 1
C1_SUB = "Real series"
C1_NOTE = "+ same on time-\nreversed series"
C1_BOX_TITLE = "Statistical break tests"
C1_BOX_ITEMS = ["two-sample tests (before vs after)",
                "likelihood-ratio tests on AR residuals"]
C1_TOKENS = ["$f_1$", "$f_2$", "$f_3$", "$f_k$"]
C1_CAPTION = "hundreds of features"
# column 2
C2_BOX_TITLE = "Synthetic-data generator"
C2_BOX_SUB = "fitted to real series · realism AUC 0.58"
C2_NOTE1 = "unlimited labelled series,\nall hidden parameters known"
C2_MODEL_TITLE = "Trained on synthetic data only"
C2_MODEL_ITEMS = ["CNN + fine-tuned Chronos-2\n(zero-shot AUC 0.84 / 0.85)",
                  "Estimators: series type,\nbreak kind and size",
                  "Noise classifier per segment"]
C2_NOTE2 = "applied to real series"
C2_TOKENS = ["$d_1$", "$d_2$", "$d_3$", "$d_m$"]
C2_CAPTION = "extra feature columns"
C2_FOOT = "no real labels used"
# column 3
C3_SUB = "Features + detector scores"
C3_IN_TOKENS = [("$f_1$", "b"), ("$f_k$", "b"), ("$d_1$", "r"), ("$d_m$", "r")]
C3_ARMS = "TabPFN / LightGBM arms"
C3_TOKENS = ["TabPFN", "LGBM", "TabPFN"]
C3_COUNT = "× 296"
C3_ARM_NOTE = "best single arm 0.873"
C3_BOX_TITLE = "TabPFN stacker"
C3_BOX_SUB = "out-of-fold scores"
C3_NOTE = "3 × stacker rank\n+ 1 × blend rank"
C3_OUT = [("OOF AUC", "0.8741"), ("Public AUC", "0.8779")]
C3_CAPTION = "P(break) per series"
# === END DATA SECTOR ===

# ---------------- style anchors ----------------
# COL_BLUE_LINE = "#0e87cc"   # L1-PIL: median of dark pixels on the series line (ref x70-530,y140-270 disp)
# COL_TOKEN_BLUE = "#7fb8eb"  # L1-PIL: 6x6 median inside a blue token (orig 71,828)
# COL_CTX_FILL = "#e5f1fb"    # L1-PIL: context-token container fill (orig 1184,159)
# COL_PINK = "#fbe6f0"        # L1-PIL: model-box fill (orig 1041,355)
# COL_PURPLE = "#874289"      # L1-PIL: model-box text / stage labels / bars
# COL_GREY_PANEL = "#f5f5f5"  # L1-PIL: quantization panel fill (orig 165,615)
# COL_RED_TOKEN = "#eab5b1"   # L1-PIL: sampled token fill (orig 1154,834)
# COL_RED_FILL = "#fbf0ef"    # L1-PIL: sampled-token container fill (orig 1870,592)
# COL_RED_LINE = "#fd5956"    # L1-PIL: forecast line dark pixels
# COL_EDGE = "#5e5e5e"        # L1-PIL: darkest-pixel median on box edges (thick 3px edges -> reliable)
# COL_SUB = "#5e5e5e"         # L1-PIL: subtitle text dark-pixel median
# COL_DIV = "#1b1b1b"         # L1 min-along-line on divider -> near-black hairline class
# COL_CONN = "#aaaaaa"        # L1 darkest pixel on thin grey connectors (light-grey guide class)
COL_BLUE_LINE = "#0e87cc"
COL_TOKEN_BLUE = "#7fb8eb"
COL_CTX_FILL = "#e5f1fb"
COL_PINK = "#fbe6f0"
COL_PURPLE = "#874289"
COL_GREY_PANEL = "#f5f5f5"
COL_RED_TOKEN = "#eab5b1"
COL_RED_FILL = "#fbf0ef"
COL_RED_LINE = "#fd5956"
COL_EDGE = "#5e5e5e"
COL_SUB = "#5e5e5e"
COL_DIV = "#1b1b1b"
COL_CONN = "#9a9a9a"
COL_TEXT = "#111111"

# Font: reference is a Helvetica-class bold grotesk -> Nimbus Sans (Helvetica clone)
_avail = {f.name for f in font_manager.fontManager.ttflist}
FAMILY = next((f for f in ["Helvetica", "Nimbus Sans", "Nimbus Sans L", "Arial",
                           "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [FAMILY, "Liberation Sans", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = FAMILY
plt.rcParams["mathtext.it"] = FAMILY + ":italic"

# Text-size ratios from L1 cap heights: title 31px : subtitle 25 : body 19 : small 14
FS_TITLE = 19
FS_SUB = 15.5
FS_MODEL = 15.5
FS_BODY = 12.5
FS_SMALL = 11.5
FS_TOKEN = 12.5

W, H = 13.33, 6.6   # content area of a 16:9 slide under a title band
DPI = 180
fig = plt.figure(figsize=(W, H), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

Z_BOX, Z_TOK, Z_LINE, Z_TEXT = 2, 3, 4, 10
LW_BOX = 2.0   # thick rounded box edges (reference ~3px at 2368 px width)


def rbox(x0, y0, w, h, fc, ec=COL_EDGE, lw=LW_BOX, r=0.12, z=Z_BOX):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(x0, y0, x1, y1, color=COL_EDGE, lw=1.6, ms=12, z=Z_LINE):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                        color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def txt(x, y, s, **kw):
    kw.setdefault("zorder", Z_TEXT)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("color", COL_TEXT)
    return ax.text(x, y, s, **kw)


def token_row(cx, y, labels, fc, container_fc, tok_w=0.62, tok_h=0.5, gap=0.42,
              stacked=False, extra=None):
    n = len(labels)
    extra_w = 0.62 if extra else 0.0
    inner = n * tok_w + (n - 1) * gap + extra_w
    cw = inner + 0.4
    ch = tok_h + 0.32
    x0 = cx - cw / 2
    rbox(x0, y - ch / 2, cw, ch, container_fc, r=0.12)
    xs = x0 + 0.2
    for i, lab in enumerate(labels):
        if stacked:
            for k in (2, 1):
                ax.add_patch(Rectangle((xs - 0.035 * k, y - tok_h / 2 + 0.035 * k), tok_w, tok_h,
                                       fc=fc, ec="#222222", lw=0.8, zorder=Z_TOK))
        ax.add_patch(Rectangle((xs, y - tok_h / 2), tok_w, tok_h, fc=fc, ec="#222222",
                               lw=0.9, zorder=Z_TOK + 0.5))
        fs = FS_TOKEN if lab.startswith("$") else FS_SMALL - 1.5
        txt(xs + tok_w / 2, y, lab, fontsize=fs)
        if i < n - 1:
            # stacked decks: next card's back layers extend 0.07 left -> centre dots in the clear gap
            back = 0.07 if stacked else 0.0
            txt(xs + tok_w + (gap - back) / 2, y - 0.02, "…", fontsize=FS_BODY, fontweight="bold")
        xs += tok_w + gap
    if extra:
        txt(xs - gap + 0.06 + extra_w / 2, y, extra, fontsize=FS_BODY + 1, fontweight="bold")
    return x0, y - ch / 2, cw, ch


# ---------------- columns ----------------
DIV = [4.30, 9.10]
CX = [2.15, 6.70, 11.22]
for xd in DIV:
    ax.plot([xd, xd], [0.05, H - 0.02], color=COL_DIV, lw=1.0, zorder=1, solid_capstyle="butt")
Y_TITLE = H - 0.32
for cx, t in zip(CX, COL_TITLES):
    txt(cx, Y_TITLE, t, fontsize=FS_TITLE, fontweight="bold", color="#000000")

# ---- column 1 ----
cx = CX[0]
txt(cx, 5.80, C1_SUB, fontsize=FS_SUB, fontweight="bold", color=COL_SUB)
gx0, gx1, gy0, gy1 = 0.35, 3.95, 4.42, 5.30
v = np.asarray(SERIES)
xs = np.linspace(gx0, gx1, len(v))
ys = gy0 + (v - v.min()) / (v.max() - v.min()) * (gy1 - gy0)
ax.plot(xs[:N0 + 1], ys[:N0 + 1], color=COL_BLUE_LINE, lw=1.6, zorder=Z_LINE,
        solid_joinstyle="round")
ax.plot(xs[N0:], ys[N0:], color=COL_RED_LINE, lw=1.6, zorder=Z_LINE, solid_joinstyle="round")
xb = xs[N0]
ax.plot([xb, xb], [gy0 - 0.08, gy1 + 0.06], color="#222222", lw=1.3, ls=(0, (3, 2)), zorder=Z_LINE + 1)
txt(xb + 0.06, gy1 + 0.17, "$n_0$", ha="left", fontsize=FS_BODY, color="#222222")

arrow(cx, 4.28, cx, 3.88)
txt(cx + 0.14, 4.08, C1_NOTE, fontsize=FS_SMALL, color=COL_PURPLE, ha="left", linespacing=1.05)
bx0, by0, bw, bh = 0.32, 2.50, 3.66, 1.32
rbox(bx0, by0, bw, bh, COL_GREY_PANEL)
txt(cx, by0 + bh - 0.32, C1_BOX_TITLE, fontsize=FS_SUB - 1, fontweight="bold")
for i, it in enumerate(C1_BOX_ITEMS):
    txt(bx0 + 0.22, by0 + bh - 0.72 - i * 0.34, "• " + it, fontsize=FS_BODY - 1, ha="left")
arrow(cx, by0 - 0.04, cx, 1.88)
c1 = token_row(cx, 1.45, C1_TOKENS, COL_TOKEN_BLUE, COL_CTX_FILL, tok_w=0.56, gap=0.36)
txt(cx, 0.80, C1_CAPTION, fontsize=FS_SUB - 2.5, fontweight="bold", color=COL_SUB)

# ---- column 2 (accent) ----
cx = CX[1]
gbx0, gby0, gbw, gbh = cx - 2.0, 4.86, 4.0, 0.98
rbox(gbx0, gby0, gbw, gbh, COL_GREY_PANEL)
txt(cx, gby0 + gbh - 0.32, C2_BOX_TITLE, fontsize=FS_SUB - 1, fontweight="bold")
txt(cx, gby0 + 0.28, C2_BOX_SUB, fontsize=FS_BODY - 1)
arrow(cx, gby0 - 0.04, cx, 4.18)
txt(cx + 0.14, 4.50, C2_NOTE1, fontsize=FS_SMALL, color=COL_PURPLE, ha="left", linespacing=1.05)
mx0, my0, mw, mh = cx - 2.05, 2.26, 4.1, 1.88
rbox(mx0, my0, mw, mh, COL_PINK, lw=LW_BOX + 0.4, r=0.14)
txt(cx, my0 + mh - 0.30, C2_MODEL_TITLE, fontsize=FS_MODEL, fontweight="bold", color=COL_PURPLE)
iy = my0 + mh - 0.62
_r = fig.canvas.get_renderer()
for it in C2_MODEL_ITEMS:
    t = txt(mx0 + 0.24, iy, "\u2022 " + it.replace("\n", "\n   "), fontsize=FS_BODY - 0.5,
            ha="left", va="top", linespacing=1.05)
    bb = t.get_window_extent(_r)
    iy -= bb.height / DPI + 0.10
arrow(cx, my0 - 0.04, cx, 1.88)
txt(cx + 0.14, 2.06, C2_NOTE2, fontsize=FS_SMALL, color=COL_PURPLE, ha="left")
c2 = token_row(cx, 1.45, C2_TOKENS, COL_RED_TOKEN, COL_RED_FILL, tok_w=0.56, gap=0.36)
txt(cx, 0.80, C2_CAPTION, fontsize=FS_SUB - 2.5, fontweight="bold", color=COL_SUB)
txt(cx, 0.50, C2_FOOT, fontsize=FS_SMALL, color=COL_PURPLE, fontstyle="italic")

# ---- column 3 ----
# Hand-off as in the reference: the column restarts with a repeated token strip
# (blue feature cards from col 1 + rose detector cards from col 2); no connector crosses a divider.
cx = CX[2]
txt(cx, 5.80, C3_SUB, fontsize=FS_SUB, fontweight="bold", color=COL_SUB)
tw, th, gp, gmid = 0.50, 0.44, 0.34, 0.46
n = len(C3_IN_TOKENS)
inner = n * tw + (n - 2) * gp + gmid
iw, ih, iy = inner + 0.4, th + 0.28, 5.20
ix0 = cx - iw / 2
rbox(ix0, iy - ih / 2, iw, ih, COL_GREY_PANEL, r=0.12)
xs_ = ix0 + 0.2
for i, (lab, kind) in enumerate(C3_IN_TOKENS):
    fc = COL_TOKEN_BLUE if kind == "b" else COL_RED_TOKEN
    ax.add_patch(Rectangle((xs_, iy - th / 2), tw, th, fc=fc, ec="#222222", lw=0.9, zorder=Z_TOK + 0.5))
    txt(xs_ + tw / 2, iy, lab, fontsize=FS_TOKEN - 0.5)
    if i < n - 1:
        g = gmid if i == n // 2 - 1 else gp
        txt(xs_ + tw + g / 2, iy - 0.02, "+" if g == gmid else "…", fontsize=FS_BODY,
            fontweight="bold")
        xs_ += tw + g
arrow(cx, iy - ih / 2 - 0.04, cx, 4.60)
c3 = token_row(cx, 4.18, C3_TOKENS, COL_RED_TOKEN, COL_RED_FILL, tok_w=0.66, tok_h=0.44,
               gap=0.38, stacked=True, extra=C3_COUNT)
txt(cx, 3.60, C3_ARMS, fontsize=FS_SUB - 2.5, fontweight="bold", color=COL_SUB)
txt(cx, 3.35, C3_ARM_NOTE, fontsize=FS_SMALL, color=COL_SUB)
arrow(cx, 3.18, cx, 2.92)
sx0, sy0, sw, sh = cx - 1.6, 2.14, 3.2, 0.74
rbox(sx0, sy0, sw, sh, COL_GREY_PANEL)
txt(cx, sy0 + sh - 0.24, C3_BOX_TITLE, fontsize=FS_SUB - 1, fontweight="bold")
txt(cx, sy0 + 0.20, C3_BOX_SUB, fontsize=FS_BODY - 1)
arrow(cx, sy0 - 0.04, cx, 1.58)
txt(cx + 0.14, 1.84, C3_NOTE, fontsize=FS_SMALL, color=COL_PURPLE, ha="left", linespacing=1.05)
ox0, oy0, ow, oh = cx - 1.55, 0.58, 3.1, 0.96
rbox(ox0, oy0, ow, oh, COL_RED_FILL)
for i, (k, val) in enumerate(C3_OUT):
    yy = oy0 + oh - 0.27 - i * 0.42
    txt(cx - 0.12, yy, k, fontsize=FS_BODY + 0.5, ha="right", color="#222222")
    txt(cx - 0.02, yy, val, fontsize=FS_SUB + 2, ha="left", fontweight="bold", color=COL_RED_LINE)
txt(cx, 0.32, C3_CAPTION, fontsize=FS_SUB - 2.5, fontweight="bold", color=COL_SUB)

fig.savefig("figure.png", dpi=DPI, facecolor="white"); fig.savefig("figure.pdf", dpi=DPI, facecolor="white"); fig.savefig("output.png", dpi=DPI, facecolor="white")


# ---------------- floor self-check ----------------
def floor_check():
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in ax.texts if t.get_text().strip()]
    bbs = [(t, t.get_window_extent(r)) for t in texts]
    over = []
    for i, (a, ba) in enumerate(bbs):
        for b, bb in bbs[i + 1:]:
            if ba.overlaps(bb):
                over.append(f"{a.get_text()!r} <-> {b.get_text()!r}")
    clip = [t.get_text() for t, b in bbs
            if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1]
    # text crossing a column divider
    inv = ax.transData.inverted()
    cross = []
    for t, b in bbs:
        x0d = inv.transform((b.x0, b.y0))[0]
        x1d = inv.transform((b.x1, b.y1))[0]
        for xd in DIV:
            if x0d < xd < x1d:
                cross.append(t.get_text())
    # text vs connector/arrow/series line geometry (sample line pixels against text boxes)
    obsc = []
    lines = [l for l in ax.lines]
    for t, b in bbs:
        for l in lines:
            xy = ax.transData.transform(np.column_stack(l.get_data()))
            # densify
            pts = []
            for p, q in zip(xy[:-1], xy[1:]):
                for s in np.linspace(0, 1, 30):
                    pts.append(p + s * (q - p))
            pts = np.array(pts) if pts else xy
            inside = ((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) & (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1))
            if inside.any():
                obsc.append(f"{t.get_text()!r} hit by line")
    # text overflowing its enclosing rounded box horizontally
    patches = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    overflow = []
    for t, b in bbs:
        cxp, cyp = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        for p in patches:
            pb = p.get_window_extent(r)
            if pb.x0 < cxp < pb.x1 and pb.y0 < cyp < pb.y1:
                if b.x0 < pb.x0 + 2 or b.x1 > pb.x1 - 2 or b.y0 < pb.y0 + 1 or b.y1 > pb.y1 - 1:
                    overflow.append(t.get_text())
    lines_out = [
        f"font_family: {FAMILY}",
        f"figure_px: {int(W * DPI)}x{int(H * DPI)}  aspect={W / H:.3f}",
        f"text_text_overlap: {'PASS' if not over else 'FAIL ' + '; '.join(over)}",
        f"text_clipped_canvas: {'PASS' if not clip else 'FAIL ' + '; '.join(clip)}",
        f"text_crosses_divider: {'PASS' if not cross else 'FAIL ' + '; '.join(cross)}",
        f"text_obscured_by_marks: {'PASS' if not obsc else 'FAIL ' + '; '.join(obsc)}",
        f"text_overflows_box: {'PASS' if not overflow else 'FAIL ' + '; '.join(overflow)}",
        f"text_zorder_above_shapes: {'PASS' if all(t.get_zorder() > Z_LINE + 1 for t in texts) else 'FAIL'}",
        "axis_labels/ticks: N/A (schematic, axes off)",
    ]
    return "\n".join(lines_out)


if __name__ == "__main__":
    rep = floor_check()
    print(rep)
    with open("floor_selfcheck_final.txt", "w") as f:
        f.write(rep + "\n")
