"""FigMirror E_noise iter 0: ridgeline density of standardized AR(5) residuals,
before vs after break, in the style of Chronos (Ansari et al. 2024) token-ID ridgeline."""
# === DATA SECTOR (edit here) ===
BEFORE = [  # before_break, n=403
    0.447,1.745,-0.449,0.266,1.206,0.366,-0.299,-0.593,-0.378,1.377,0.425,-2.317,-0.648,-0.234,-0.050,0.816,-1.306,1.053,1.932,1.996,
    -0.362,-0.152,0.039,0.026,0.832,0.147,0.006,0.869,-1.166,-1.029,-0.024,2.292,1.624,-0.381,0.075,-1.613,-0.065,-0.269,-1.152,-1.352,
    -0.887,-0.686,-0.022,0.462,1.224,-2.249,-0.216,0.235,0.783,0.070,1.312,-0.828,-1.624,-0.514,1.576,-0.316,-1.052,-0.996,0.178,-0.120,
    0.382,-0.630,-1.240,-0.116,-0.986,0.523,-1.729,0.485,0.454,-0.927,-1.364,0.615,-0.961,-0.472,1.215,-1.123,-0.859,-0.324,-0.977,0.778,
    -0.394,-0.150,0.542,-0.959,-0.899,-1.028,0.211,-1.895,0.775,1.233,-0.662,0.477,0.713,0.906,-0.571,-0.272,-0.346,0.829,-0.560,-0.959,
    0.428,-0.113,-2.427,-1.338,-0.807,1.160,0.936,1.379,-0.852,-0.814,0.228,-2.008,0.245,-0.598,1.018,-0.485,-0.632,1.717,1.238,0.878,
    1.475,-0.719,1.323,-0.166,-0.160,0.739,1.523,-0.824,1.097,-0.119,0.714,-0.326,0.936,-0.033,-0.407,0.835,-0.971,-0.380,-1.033,0.380,
    1.566,0.676,1.074,-0.521,0.384,-1.701,0.376,0.321,-0.102,-0.823,-0.450,-2.441,-0.127,0.262,-0.881,-0.100,-0.804,1.487,-0.465,-0.733,
    -0.944,1.574,-0.380,1.853,-0.367,0.506,0.480,0.826,2.725,0.876,-0.944,0.388,2.196,-1.303,0.472,-1.921,-0.880,0.907,-0.578,0.419,
    -0.314,-1.536,-0.684,0.881,0.367,-1.031,0.025,0.614,1.272,-0.887,0.770,-1.350,-0.779,-0.783,-1.399,0.691,-1.795,-1.365,-0.076,0.174,
    -0.403,-0.214,1.110,-0.639,-1.283,-2.613,0.075,-0.707,-0.401,1.099,-0.659,1.579,-0.357,-0.787,-0.233,-0.793,1.408,-0.431,1.560,0.305,
    -0.012,-1.348,-0.320,0.093,-0.111,1.196,0.303,0.310,-0.264,0.487,0.272,1.845,0.335,0.985,-0.393,0.782,0.150,2.074,-1.528,1.681,
    -1.198,-0.425,0.454,-0.058,-0.089,-1.018,1.371,-0.072,0.280,1.332,-0.331,0.998,0.369,-1.845,0.511,-1.267,-1.926,0.980,0.642,0.863,
    0.223,0.757,0.059,-0.183,-0.038,0.178,1.329,0.367,1.355,0.479,0.141,0.376,1.905,0.127,0.104,-1.238,-0.740,1.181,-0.885,-1.481,
    -0.154,0.106,0.714,-1.166,-0.012,-1.357,0.242,1.129,0.366,-0.546,1.862,-0.998,-0.118,0.992,-0.722,1.338,-0.973,-0.046,-0.892,0.561,
    -0.122,1.424,-2.206,0.445,-0.131,0.503,0.220,0.952,-0.258,0.720,-1.226,-0.706,0.466,1.440,2.017,-0.351,0.216,-1.325,-0.024,0.287,
    1.264,-1.331,-1.405,-0.149,0.228,1.068,-0.306,-2.716,-0.669,-0.967,1.125,0.570,1.174,1.782,0.106,0.609,0.551,0.980,0.263,0.481,
    -1.337,-0.029,0.977,-0.373,0.708,0.039,-0.503,0.547,-0.639,-1.707,0.840,1.011,-1.485,1.225,0.728,-2.631,0.108,-0.246,-0.438,0.142,
    -1.268,1.699,0.415,-0.749,1.300,1.644,0.696,0.721,1.231,0.034,-2.287,-0.227,-0.979,-1.041,0.059,1.035,0.916,0.123,0.337,-0.649,
    0.748,0.601,-0.980,-0.182,0.846,-1.463,-0.008,1.756,1.964,-0.646,0.615,0.743,-1.588,0.372,0.762,-0.136,-3.443,-0.124,-0.067,-0.726,
    -1.020,0.731,-0.269,
]
AFTER = [  # after_break, n=534
    0.681,-0.402,1.070,0.682,1.055,-0.310,0.378,-0.460,1.235,1.180,-1.190,0.912,-1.205,0.038,1.202,0.040,1.804,-0.742,0.901,-0.601,
    0.436,-3.010,-0.004,-0.192,0.692,0.269,-0.187,-1.149,0.317,1.798,-1.050,-0.919,0.536,1.035,-1.993,0.392,2.700,-0.240,0.694,-2.952,
    -0.901,-0.831,-0.390,-1.075,-0.793,1.542,0.831,2.151,0.404,-0.096,-0.134,-0.465,-0.589,-0.213,-0.709,-0.955,0.083,-0.545,0.106,2.862,
    0.473,1.140,0.118,-0.678,0.142,1.537,1.375,-1.621,-0.692,0.976,-0.457,-0.990,0.666,0.339,-2.217,0.593,-0.440,-0.676,-1.020,0.280,
    -1.761,0.686,0.506,0.603,0.250,0.859,0.006,0.747,-0.284,-0.858,0.146,0.471,-0.072,-1.058,0.685,1.236,0.026,0.142,0.089,-0.204,
    -1.352,-1.155,-0.575,-1.850,-1.101,0.881,-0.591,-0.745,0.511,-1.661,-0.347,0.087,-0.258,-1.034,3.195,-0.129,2.297,0.321,-0.390,2.319,
    -3.684,0.362,-1.162,0.057,1.167,-0.492,-0.768,-0.139,0.132,-0.225,1.337,0.403,1.337,0.373,-1.329,0.012,1.517,-0.085,0.846,1.790,
    -1.190,-0.484,-0.939,-0.658,0.397,-0.542,-0.128,0.132,0.426,0.610,1.520,0.046,-0.044,1.092,-0.133,0.350,-1.983,-0.326,1.416,0.573,
    0.128,0.294,1.171,1.082,0.678,0.698,2.216,6.588,0.817,0.004,1.169,-0.804,-0.010,-0.029,0.743,-0.321,1.380,0.225,-1.622,-0.598,
    -1.600,-0.236,-1.428,-1.146,0.661,-0.877,1.100,-1.659,-1.383,0.203,0.446,-0.028,1.020,0.278,1.831,-0.103,0.736,1.379,0.193,-1.172,
    -0.103,-1.870,0.611,0.458,0.175,-0.898,-0.402,0.912,-0.771,0.192,-0.000,-2.692,0.972,0.759,-0.272,-0.255,-0.242,-1.013,-1.478,-1.299,
    0.045,1.162,-0.480,0.167,-0.728,1.447,0.406,-0.659,-0.028,-0.189,-0.012,-0.520,-0.160,0.220,0.405,-0.252,0.252,-0.533,-1.056,1.919,
    -1.021,1.504,0.506,-1.886,1.027,0.135,0.194,-0.601,0.650,1.198,-1.541,1.164,-1.906,-0.232,-0.535,-1.447,1.026,0.740,0.841,0.553,
    1.924,-0.244,-0.037,-0.642,-1.495,-0.334,-0.315,-1.704,-1.050,0.194,-0.748,0.811,-0.614,0.056,0.918,0.585,-2.066,1.991,0.085,1.940,
    -0.069,-1.187,0.220,-0.048,0.586,-0.482,0.410,0.553,0.435,-0.392,-0.219,-0.275,0.176,-0.636,-1.166,0.037,-0.638,-0.091,0.567,-0.622,
    1.378,-0.665,0.345,0.292,-1.266,-0.074,-0.216,-0.578,-0.226,-0.088,0.305,0.624,-0.718,-2.078,-0.820,0.718,0.105,0.849,0.612,0.592,
    0.388,-1.060,-1.506,0.551,-0.016,-0.313,0.206,0.734,-0.949,-1.486,1.607,0.220,0.663,-0.251,0.597,0.001,-0.338,1.313,-1.113,0.070,
    0.133,-0.376,-1.037,-0.557,-0.135,-0.532,0.315,-0.479,-0.511,0.175,-0.701,0.525,-1.173,-0.360,2.121,0.035,-0.153,0.294,1.891,1.420,
    -0.859,-0.337,0.572,0.003,-0.604,1.715,-1.171,-0.524,1.232,0.424,-0.274,0.866,0.430,-0.525,0.521,-0.974,-0.126,-0.864,-0.342,0.185,
    -0.402,0.085,-0.641,-0.045,-1.279,-0.019,0.439,0.116,0.552,-1.144,-0.453,0.667,-1.006,-1.135,0.523,-0.320,-0.437,0.673,0.632,0.630,
    -0.167,-0.062,0.280,-0.985,0.295,0.819,-2.255,-2.151,-0.777,1.403,-0.836,-0.085,0.460,0.787,0.457,-0.284,0.326,-1.083,-0.250,0.201,
    -0.257,0.655,-0.380,0.813,0.313,-1.601,0.799,0.760,-0.563,1.145,-0.683,-0.228,0.169,0.665,0.870,1.301,0.504,-0.218,-0.669,-0.079,
    -0.831,-0.201,0.708,0.315,0.152,-0.392,-2.075,0.235,-0.634,1.114,1.083,0.339,0.519,-0.328,0.057,1.246,-0.198,0.440,-1.147,-0.464,
    -2.298,-0.859,-0.835,0.183,-0.668,0.010,-0.503,0.823,0.059,-0.132,1.591,-0.810,1.424,-0.773,0.591,-0.645,-1.850,0.076,-1.305,0.958,
    -0.165,1.645,-0.424,-1.211,-2.525,-1.743,-1.824,1.640,0.099,-0.716,-0.109,-1.054,1.316,-0.405,0.921,1.003,0.235,-0.584,-0.708,-0.128,
    1.830,0.918,-0.443,-1.161,0.202,-0.353,0.246,-0.371,-0.959,0.683,1.546,-0.448,0.325,-0.226,0.645,1.850,0.092,-0.442,0.857,-1.203,
    -1.090,0.039,1.519,-0.213,0.132,-0.376,-1.651,1.897,1.157,1.656,-0.739,0.945,0.473,-0.854,
]
LABELS = ["Gaussian $\\mathcal{N}(0,1)$", "Before break", "After break"]
NOTES = [
    "beyond \u00b13 sd: 0.27%\nexcess kurtosis: 0",
    "beyond \u00b13 sd: 0.25%\nexcess kurtosis: \u22120.05",
    "beyond \u00b13 sd: 0.75%\nexcess kurtosis: 3.58",
]
XLABEL = "Standardized residual"
XTICKS = [-6, -3, 0, 3, 6]
# === END DATA SECTOR ===

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
# Font: L1-perceived. Tick digits / axis label are Helvetica/Arial-class grotesk
# -> Liberation Sans (Arial metric clone). Ridge labels "h = 1" are DejaVu Sans
# mathtext glyphs -> DejaVu Sans + dejavusans mathtext.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"

# --- colours ---
# Ridge ramp, L1-PIL: median of 6x6 window at ridge interiors (orig px):
#   h1 (480,96) #b8b889 | h3 (1152,408) #5f7b3e | h5 (1296,720) #173114
# Our 3 ridges take the ramp endpoints + middle (light -> dark, top -> bottom).
RIDGE_COLS = ["#b8b889", "#5f7b3e", "#173114"]
COL_TEXT = "#262626"   # L1-PIL: darkest pixel of tick glyph "2150" = (38,38,38)
COL_GUIDE = "#c8c8c8"  # L2-class: dashed light-grey guide (only for the +-3 sd marks; not in ref)

# --- geometry (L1-measured ratios, recomputed for our figure) ---
FIGSIZE = (12.0, 4.9)          # ref aspect 2400/979 = 2.45 -> 2.45
DPI = 200
SPACING = 1.0                  # ridge baseline spacing (data units)
PEAK_H = 1.04 * SPACING        # ref: h=1 peak ~1.04 x spacing above its baseline
BASE_LW = 4.0                  # ref baseline ~11 px of 979 px -> ~4 pt at 4.9 in
HALO_LW = 5.0                  # white outline around each ridge curve (ref ~4 px each side)
TICK_FS = 22                   # ref digit cap height 45 px / 979 px
AXLAB_FS = 25                  # ref "Token ID" cap 50 px
RIDGE_FS = 25                  # ref "h = 1" glyph band 53 px
NOTE_FS = 17

XLO, XHI = -6.4, 7.2

def kde(sample, grid):
    s = np.asarray(sample, float)
    bw = 1.06 * s.std(ddof=1) * len(s) ** (-1 / 5)   # Silverman/Scott-style bandwidth
    z = (grid[:, None] - s[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(1) / (len(s) * bw * np.sqrt(2 * np.pi))

grid = np.linspace(XLO, XHI, 1600)
dens = [np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi), kde(BEFORE, grid), kde(AFTER, grid)]
scale = PEAK_H / dens[0].max()   # common scale: same area for every ridge, so peaks compare honestly

fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
ax = fig.add_axes([0.0, 0.205, 1.0, 0.78])
n = len(LABELS)
bases = [(n - 1 - i) * SPACING for i in range(n)]

for i, (lab, d, c, b) in enumerate(zip(LABELS, dens, RIDGE_COLS, bases)):
    z0 = 2 + 3 * i
    y = b + d * scale
    m = d * scale > 0.004
    yh = np.where(m, y, np.nan)
    ax.plot(grid, yh, color="white", lw=HALO_LW, solid_capstyle="round", zorder=z0)
    ax.fill_between(grid, b, y, color=c, lw=0, zorder=z0 + 0.5)
    ax.axhline(b, color=c, lw=BASE_LW, zorder=z0 + 1, solid_capstyle="butt")
    # ridge label: left edge, in ridge colour, sitting on the baseline (ref motif)
    ax.text(XLO + 0.02, b + 0.10, lab, color=c, fontsize=RIDGE_FS, ha="left", va="bottom",
            family="DejaVu Sans", zorder=50)
    # tail statistics on the empty right side of each ridge
    ax.text(XHI - 0.05, b + 0.10, NOTES[i], color=COL_TEXT, fontsize=NOTE_FS,
            ha="right", va="bottom", linespacing=1.15, zorder=50)

# +-3 sd guides behind ridges (pointing out the tail region)
for xv in (-3, 3):
    ax.axvline(xv, color=COL_GUIDE, lw=1.0, ls=(0, (4, 3)), zorder=1)

ax.set_xlim(XLO, XHI)
ax.set_ylim(-0.06, bases[0] + PEAK_H + 0.04)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_yticks([])
ax.set_xticks(XTICKS)
ax.set_xticklabels([("−" if t < 0 else "") + str(abs(t)) for t in XTICKS])
ax.tick_params(axis="x", length=0, pad=9, labelsize=TICK_FS, colors=COL_TEXT)
ax.set_xlabel(XLABEL, fontsize=AXLAB_FS, color=COL_TEXT, labelpad=6)
for t in ax.get_xticklabels():
    t.set_zorder(50)
ax.patch.set_alpha(0)

# ---------------- floor self-check ----------------
fig.canvas.draw()
r = fig.canvas.get_renderer()
texts = [t for t in ax.get_xticklabels() if t.get_text()] + list(ax.texts) + [ax.xaxis.label]
bbs = [(t.get_text(), t.get_window_extent(r)) for t in texts]
lines = []
ov = [f"{a} <-> {b}" for i, (a, ba) in enumerate(bbs) for b, bb in bbs[i + 1:] if ba.overlaps(bb)]
lines.append("text_overlap: " + ("PASS" if not ov else "FAIL " + "; ".join(ov)))
fb = fig.bbox
cl = [a for a, bb in bbs if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5]
lines.append("clipped_text: " + ("PASS" if not cl else "FAIL " + "; ".join(cl)))
xl = ax.xaxis.label.get_window_extent(r)
lines.append(f"xlabel_y0_px: {xl.y0:.1f} -> " + ("PASS" if xl.y0 >= 0 else "FAIL"))
# text vs ridge marks: in-axes texts must not intersect any ridge fill above its own label region
inv = ax.transData.inverted()
obs = []
for t in ax.texts:
    bb = t.get_window_extent(r)
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    sel = (grid >= x0) & (grid <= x1)
    for d, b in zip(dens, bases):
        top = b + d[sel] * scale
        # a fill from b..top intersects the text box if b < y1 and top > y0
        if b < y1 and (top > y0 + 1e-9).any() and (top - b > 0.01).any():
            obs.append(f"'{t.get_text()[:20]}' vs ridge@{b}")
max_mark_z = max(a.get_zorder() for a in ax.get_children() if not isinstance(a, Text))
lowz = [t.get_text() for t in texts if t.get_zorder() <= max_mark_z and t is not ax.xaxis.label]
lines.append("text_obscured_by_marks: " + ("PASS" if not obs and not lowz else "FAIL " + "; ".join(obs + lowz)))
lines.append(f"figure_aspect: {FIGSIZE[0]/FIGSIZE[1]:.3f} (ref 2.451, band 2.21-2.70)")
lines.append("top_row_xlabel: n/a (single axes)")
fig.savefig("figure.png", dpi=DPI, facecolor="white"); fig.savefig("figure.pdf", dpi=DPI, facecolor="white"); fig.savefig("output.png", dpi=DPI, facecolor="white")
with open("floor_selfcheck_final.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
