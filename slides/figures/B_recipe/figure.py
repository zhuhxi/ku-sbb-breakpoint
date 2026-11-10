"""FigMirror B_recipe iter 2 -- KU-SBB synthetic-data generator diagram,
in the visual style of the Chronos KernelSynth figure (Ansari et al., 2024)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
# Helvetica-class sans (L1-perceived: reference type is Helvetica Neue-like)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Nimbus Sans"
plt.rcParams["mathtext.it"] = "Nimbus Sans:italic"

# === DATA SECTOR (edit here) ===
WHITE_NOISE = [0.586,2.096,1.602,0.321,1.239,0.305,-0.461,1.040,1.348,-2.501,1.010,1.701,-0.194,1.237,1.625,0.418,2.021,2.591,3.899,1.883,0.345,0.681,1.857,1.200,0.244,1.736,-0.557,1.226,2.495,-0.084,0.952,1.022,1.240,0.117,0.749,1.755,0.801,1.272,0.452,2.406,0.009,0.913,-0.320,1.619,-0.281,0.724,1.056,1.174,0.763,1.591,0.765,0.235,0.874,2.904,1.178,1.857,1.169,0.034,0.664,1.043,-0.888,1.114,-0.113,2.358,1.828,-1.315,0.434,3.275,0.663,2.078,2.336,0.601,0.570,1.698,0.724,-0.404,1.978,1.457,1.319,1.716,2.538,2.663,0.861,1.535,0.777,0.549,1.959,-0.986,1.282,0.012,0.456,1.198,1.248,1.495,-0.019,0.785,1.091,2.865,2.914,0.768,0.511,0.999,0.554,2.714,0.534,0.768,1.655,0.590,2.057,1.502,0.081,1.388,-0.423,0.938,-0.090,2.198,1.389,2.226,1.397,1.386,1.636,1.813,2.611,0.042,1.058,-0.052,1.473,0.658,3.408,2.443,0.528,0.590,-0.196,1.491,3.179,-0.496,1.210,1.392,1.570,0.948,1.417,0.616,-0.686,-0.285,0.876,0.900,1.866,1.514,1.686,2.315,0.690,3.318,1.318,1.706,1.499,0.430,1.304,0.009,0.705,1.193,2.065,3.895,0.656,0.516,0.007,2.374,0.536,0.290,0.831,0.339,1.744,1.473,1.113,0.971,0.706,0.187,2.248,0.041,0.606,2.276,2.100,1.078,2.748,4.459,0.847,0.908,0.842,0.697,1.483,1.462,0.640,0.541,0.913,0.445,0.494,0.860,2.227,0.854,1.713,3.448]
AR1 = [0.031,0.650,1.805,2.185,1.924,0.802,-0.094,-0.007,4.398,2.567,0.413,1.962,1.382,1.462,1.947,0.248,0.438,2.108,0.628,0.518,1.659,2.207,1.463,1.025,1.054,1.564,-0.005,0.693,-0.631,0.235,0.635,0.699,1.803,1.632,-0.457,0.987,1.966,2.200,2.047,0.584,0.230,-0.789,0.463,3.499,3.266,1.856,1.247,1.062,1.039,-3.789,-1.108,-0.554,-0.725,0.096,-0.305,-1.916,-2.967,-1.443,-1.516,0.214,-0.837,1.048,2.244,0.534,1.393,0.391,0.784,1.026,0.996,2.084,0.913,1.392,1.699,2.582,0.593,1.258,1.320,0.470,1.610,-0.037,-0.501,-2.453,-0.709,0.705,2.749,2.131,0.633,-0.873,1.722,0.479,0.422,1.248,1.189,1.847,1.938,1.216,1.740,2.779,0.873,0.782,0.202,0.433,0.907,0.813,0.379,1.115,0.469,2.026,1.728,2.093,1.579,1.157,0.928,-1.504,-1.085,-0.542,-0.114,2.127,2.082,1.313,1.059,-0.174,0.911,2.331,1.972,3.058,4.715,1.885,0.644,0.298,-0.226,0.117,0.091,0.613,1.384,-0.204,-0.213,-0.214,1.066,-0.339,0.399,-0.102,-0.671,0.218,0.868,1.606,1.480,1.087,0.754,1.527,1.735,-0.189,0.985,0.960,1.666,1.602,2.661,1.627,1.518,4.800,2.944,1.576,4.013,1.875,0.643,1.472,1.396,1.311,1.201,1.551,2.195,2.785,1.166,-0.140,0.836,-0.160,-0.784,-0.129,1.963,2.765,2.575,1.776,2.898,1.687,1.433,0.547,-0.440,1.049,0.982,0.355,-0.109,-0.516,0.738,0.896,0.399,1.831,2.585,3.096,0.637,0.357]
SINE = [0.842,0.879,0.879,1.883,2.724,2.153,3.254,2.674,3.312,3.322,2.503,1.965,1.584,1.830,1.684,0.728,0.790,0.148,1.054,0.440,0.123,1.696,1.352,2.985,0.920,3.833,2.588,3.467,2.565,3.514,2.971,2.261,1.454,1.968,1.843,1.213,0.859,0.192,0.190,-0.162,0.856,-0.043,1.002,2.080,2.298,2.341,1.475,3.106,2.950,2.360,1.877,2.773,1.773,1.124,1.673,1.102,0.328,0.468,1.414,0.250,0.553,1.287,0.666,2.693,2.080,2.828,3.980,3.367,3.400,4.754,2.120,2.570,1.717,2.096,2.244,0.265,0.744,1.201,0.165,0.965,0.498,0.588,1.311,2.183,1.933,1.900,2.870,3.275,3.875,2.271,3.108,2.370,1.921,2.955,1.330,0.432,1.477,0.787,0.299,-0.327,1.032,0.819,1.848,1.948,2.296,2.855,3.522,2.899,3.504,3.555,2.574,2.310,2.920,1.201,2.275,1.157,0.576,0.291,0.402,1.953,0.445,0.691,1.251,2.320,2.184,2.319,2.592,2.666,3.408,2.846,2.334,1.850,2.774,1.515,1.996,1.076,0.264,1.040,0.581,0.635,1.149,0.261,1.341,1.545,1.861,2.664,2.736,3.483,3.087,3.104,2.343,2.631,1.840,1.366,0.674,1.157,0.577,0.398,0.154,0.768,0.838,1.183,1.747,1.594,2.269,2.338,2.596,2.925,2.736,3.609,2.269,1.896,1.741,1.858,1.107,1.750,0.008,1.225,1.324,1.127,0.894,0.261,2.561,2.147,2.049,2.848,4.259,3.007,3.091,3.747,3.101,2.468,1.585,1.123,1.586,0.720,0.668,0.516,-0.568,0.849]
RANDOM_WALK = [-0.481,-0.244,0.174,0.135,-0.531,-0.355,0.119,0.276,-0.452,-0.910,-0.940,-0.878,-1.393,-1.935,-2.129,-3.058,-3.160,-2.877,-2.232,-1.976,-1.419,-0.497,-0.815,-1.637,-2.396,-2.553,-2.184,-2.425,-2.106,-2.323,-2.540,-1.193,-1.377,-0.426,-0.474,-1.118,-0.678,-0.875,-0.377,0.058,-0.288,-0.304,-0.750,-0.541,0.106,-0.949,-0.794,-0.926,-1.925,-2.064,-1.150,-0.910,-0.159,0.503,0.652,0.282,1.044,-0.019,-0.066,-0.830,-0.375,-0.334,-0.619,-1.454,-2.032,-2.161,-2.097,-2.846,-3.055,-3.533,-1.477,-1.126,-1.075,-1.451,-0.862,-1.144,-1.242,-1.245,-2.016,-2.664,-3.230,-3.767,-3.081,-3.589,-3.327,-3.406,-3.005,-2.551,-2.824,0.206,-0.224,-0.882,0.257,1.268,1.541,0.964,0.122,0.202,-0.151,-0.373,-0.571,-0.655,-0.958,-1.123,-0.985,0.033,-0.557,-0.779,-1.630,-1.543,-1.497,-1.101,-1.022,-0.483,-0.484,0.291,-0.446,0.791,-0.231,-0.659,-1.573,-2.125,-2.436,-2.801,-2.630,-2.632,-2.272,-2.022,-1.741,-1.849,-1.314,-0.471,-0.279,-0.498,-1.173,-0.620,-0.465,-1.010,-1.003,-1.119,-0.858,-1.073,-0.282,-0.736,-0.009,-0.182,-0.502,-1.632,-0.899,-0.614,-0.763,-0.694,-0.428,-0.674,-1.045,-0.902,-0.438,-0.159,0.208,-0.885,-0.888,-0.299,-0.637,-0.943,-0.081,-0.298,-0.005,-0.308,-0.522,-0.251,-0.467,-0.023,0.763,1.322,1.304,0.698,0.391,-0.296,-0.118,-1.006,-0.785,-1.274,-1.569,-1.728,-1.449,-1.139,-1.793,-1.500,-0.649,-0.650,-1.203,-1.477,-1.306,-2.088,-2.306,-2.481,-2.009,-2.453,-2.116,-1.614]
OUTPUT_N0 = 228
OUTPUT = [-1.518,1.505,-0.614,0.459,-1.814,0.101,0.765,0.639,-0.425,-0.850,0.207,0.944,-0.086,0.486,-0.133,2.352,-0.393,-0.306,1.123,2.081,0.835,-0.051,0.417,0.618,0.345,1.037,0.249,0.190,-0.678,0.795,-1.612,-0.810,1.225,0.091,-0.759,0.711,0.864,1.254,1.664,0.089,0.107,0.237,1.343,-0.190,2.271,-0.857,-1.855,0.245,0.792,2.234,-0.631,0.572,-0.402,-0.715,0.510,1.110,-1.517,1.563,1.360,0.494,1.385,-0.746,-0.476,1.716,-0.020,-1.273,-0.012,-2.241,1.709,1.143,-0.279,1.971,1.841,0.031,1.892,-0.472,0.998,0.090,1.195,-1.200,-1.103,-1.958,-1.255,0.785,1.204,0.816,0.010,-1.050,1.757,0.523,0.231,1.478,0.435,2.041,0.855,0.147,-1.007,-0.339,0.517,1.392,-0.288,1.747,0.475,-0.678,-2.097,-0.031,0.749,0.098,0.167,-0.228,0.865,1.189,0.262,1.605,-0.197,2.311,0.258,-0.186,0.823,0.321,1.242,-0.817,0.896,-0.923,0.363,1.481,1.233,-0.373,-0.040,0.280,0.682,0.034,-1.090,0.457,1.652,-0.545,0.724,0.335,0.346,-0.442,-0.720,-1.273,0.256,-0.426,0.864,1.817,1.153,-2.674,1.173,-0.708,0.363,-0.414,2.126,-0.060,1.725,-1.960,-0.050,0.366,-0.928,-1.063,-0.176,0.315,-0.509,1.433,-0.077,2.116,0.486,0.930,2.302,-0.418,-0.283,1.135,0.403,-0.681,-0.500,0.862,-0.395,-0.163,-0.830,-0.238,-0.397,-0.540,0.856,0.729,0.633,1.362,0.017,0.183,0.915,0.124,-0.395,0.152,0.314,1.619,0.365,-0.331,-2.476,0.902,0.462,-0.799,0.229,0.806,0.234,1.389,-0.242,-0.017,1.293,-1.560,1.531,1.248,0.495,1.920,2.347,0.833,-0.059,-0.919,0.141,-0.313,-0.828,0.227,-0.542,1.236,0.424,1.816,0.553,0.812,-0.867,0.860,-1.190,-1.595,-0.473,0.036,-1.890,0.765,-1.359,-1.340,-0.351,-1.536,0.609,0.929,0.399,-0.608,-2.400,-1.742,0.023,-0.997,0.067,-1.028,-1.137,-1.159,-1.037,-0.273,-0.232,0.254,0.445,-0.681,-0.484,-1.081,0.735,0.073,-0.865,-1.223,-0.883,-3.357,-0.457,-1.275,1.132,-0.703,1.228,-0.392,-0.817,-0.629,-0.622,2.166,1.410,-1.411,-1.383,-0.072,-1.090,-0.618,-0.594,-1.365,-0.888,-1.448,-2.053,-1.016,-2.537,0.776,-0.838,0.641,-0.474,-0.278,0.238,-0.138,1.321,-2.267,0.800,0.228,-0.073,-0.662,-0.285,-1.305,0.116,-1.576,-1.372,0.424,-0.770,-0.810,-1.367,-2.172,-1.177,0.121,-2.191,0.443,-0.031,-1.329,-0.036,-0.998,-0.992,-0.914,-1.117,-0.084,-2.023,-0.339,-0.322,0.304,-0.461,-1.208,0.991,0.272,1.054,-0.174,1.014,-1.355,1.266,0.717,-0.029,-0.037,-0.078,-1.993,-0.477,0.086,-1.375,-3.926,0.560,-2.657,-0.217,1.777,-0.893,-0.989,-0.658,-0.461,-1.309,0.853,-0.581,-0.725,2.188,-0.482,-1.225,-0.123,-0.310,-3.056,1.729,-0.241,-0.716,-1.652,1.034,1.079,-1.207,0.752,0.012,-1.128,0.103,0.064,0.136,-0.019,-0.758,2.137,-1.090,1.443,-0.995,-1.300,0.227,-0.758,-1.586,-0.015,-0.041,0.656,-3.637,0.958,-1.727,-0.607,0.311,-0.320,-1.182,2.601,-1.457,-0.216,-0.576,-0.058,-0.357,-0.637,-0.477,0.676,-0.650,-1.331,0.322,-2.817,-0.382,1.936,-0.280,0.864,-0.937,2.172,-1.812,-1.134,0.821,0.496,-0.231,-0.325,-0.772,-2.347,-1.473,-1.442,-1.841,-0.799,0.858,-0.741,-1.397,-0.285,0.059,-1.472,0.514,-1.566,-0.568,-0.030,-0.750,-0.507,-0.039,0.229,1.280,-1.342,-0.487,1.125,-2.217,-1.529,-0.098,0.448,-1.346,-1.123,-0.939,0.082,0.982,0.950,-1.104,-1.111,1.369,1.266,-0.456,-0.043,-0.761,-1.235,-0.300,-1.776,-2.328,-0.774,-0.640,2.238,-1.000,-1.171,0.591,1.239,0.912,-0.512,-0.900,-0.958,-1.211,-1.383,-0.422,-1.587,-1.476,-1.403,-0.212,0.332,-1.373,-0.101,0.121,-0.282,-0.482,-0.045,0.571,0.562,-1.313,-0.970,0.435,-0.859,-0.964,0.499,-1.895,-2.308,-1.279,-0.953,-1.882,-1.221,0.155,0.049,-0.215,-0.978,1.532,-2.470,-0.054,-1.493,1.387,0.462,-2.260,-2.271,-0.117,0.869,-2.294,-0.003,-3.165,1.586,-0.152,-0.463,-0.582,0.292,-0.191,1.395,-1.055,-1.490,-1.539,-0.246,-2.088,1.525,-0.598,0.304,-0.385,-0.085,-1.615,-0.566,-0.394,-0.109,-2.087,1.443,0.196,-0.663,0.440,2.301,-0.882,-1.495,-0.157,0.303,-1.714,-0.714,-0.570,-2.223,-0.485,-1.204,1.201,-1.094,-0.206,-1.942,-0.128,-0.290,-1.964,-0.179,0.703,1.380,-1.077,0.382,-0.979,-0.029,0.209,-0.832,-1.198,-0.864,1.172,3.961,0.405,-0.226,-0.378,1.216,0.422,-0.802,-0.376,-1.207,-2.695,-0.048,0.132,-1.237,-1.656,-2.941,-0.610,-1.668,-1.242,-0.938,-0.513,-1.010,-1.484,-0.068,0.266,-2.161,-0.267,0.685,-1.061,-2.338,1.202,1.100,0.093,-0.772,-1.579,-1.085]
STRUCTURES = [("White noise", WHITE_NOISE), ("AR(1)", AR1),
              ("Sine", SINE), ("Random walk", RANDOM_WALK)]
NOISES = ["Gaussian", "Student-t", "Bimodal"]          # analytic densities
BREAKS = ["level", "variance", "dependence", "noise shape"]  # schematics
# === END DATA SECTOR ===

# ---- colours ----
COL_GREY = "#5e5e5e"   # L1-PIL: dominant dark pixel of box border / title / labels / connectors = (94,94,94)
COL_INK = "#000000"    # L1-PIL: kernel-thumbnail line pixels = (0,0,0)
COL_FRAME = "#d6d6d6"  # L1-PIL min-along-line of thumbnail frame: 1-2 px at (208..222)
COL_BLUE = "#0e87cc"   # L1-PIL: blue sample line, mode (14,135,204)
COL_RED = "#fd5956"    # L1-PIL: red sample line, mode (253,89,86)

# ---- geometry: canvas units, 1 unit identical in x and y ----
CW, CH = 1859.0, 1000.0              # ref aspect 2387/1284 = 1.859 (L1-PIL)
FIG_W = 16.0
FIG_H = FIG_W * CH / CW
DPI = 150
FS_LABEL = 16          # regular grey labels
FS_TITLE = 18          # bold box titles; ratio 1.13 = ref cap height 35/31 px
FS_SMALL = 13
LW_BOX = 3.2           # ref border 9 px on 2387 px canvas, scaled by thumb ratio
LW_INK = 1.3
THUMB = 140
PITCH = 190
TOP = 900
BOX_HALF = 93

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
cv = fig.add_axes([0, 0, 1, 1])
cv.set_xlim(0, CW); cv.set_ylim(0, CH); cv.axis("off")
cv.set_zorder(0)


def thumb(cx, top, w=THUMB, h=THUMB):
    ax = fig.add_axes([(cx - w / 2) / CW, (top - h) / CH, w / CW, h / CH])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(COL_FRAME); s.set_linewidth(0.8)
    ax.set_zorder(2)
    return ax


def label(x, y, s, fs=FS_LABEL, bold=False, color=COL_GREY):
    return cv.text(x, y, s, ha="center", va="center", fontsize=fs, color=color,
                   fontweight="bold" if bold else "normal", zorder=10)


def box(cx, y0, y1, dashed=False):
    p = FancyBboxPatch((cx - BOX_HALF, y0), 2 * BOX_HALF, y1 - y0,
                       boxstyle="round,pad=0,rounding_size=26",
                       fc="none", ec=COL_GREY, lw=LW_BOX,
                       ls=(0, (2.2, 1.5)) if dashed else "-",
                       capstyle="butt", zorder=1)
    cv.add_patch(p)


def arrow(p0, p1):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=1.0,head_width=0.5",
                        mutation_scale=11, color=COL_GREY, lw=LW_BOX,
                        shrinkA=0, shrinkB=0, joinstyle="miter", zorder=3)
    cv.add_patch(a)


def plot_series(ax, y, color=COL_INK, lw=LW_INK, pad=0.08):
    y = np.asarray(y, float)
    ax.plot(np.arange(len(y)), y, color=color, lw=lw, solid_joinstyle="round")
    ax.set_xlim(-0.03 * len(y), 1.03 * len(y))
    lo, hi = y.min(), y.max(); r = hi - lo
    ax.set_ylim(lo - pad * r, hi + pad * r)


def slot_top(i):
    return TOP - i * PITCH


def slot_label_y(i):
    return slot_top(i) - THUMB - 25


P_BANK = 400          # iter2: one uniform stage pitch for all 3 gaps (gutter 237 = 1.7 tile)
P_OUT = P_BANK        # iter2: result follows Break at the same stage pitch (no dead corridor)
X_S = (CW - (2 * P_BANK + P_OUT + BOX_HALF + THUMB / 2)) / 2 + BOX_HALF  # centre content
X_N, X_B = X_S + P_BANK, X_S + 2 * P_BANK
BOX_Y0, BOX_Y1 = 22, 928
TITLE_Y = 962

# ---------- Stage 1: structure bank (solid rounded box) ----------
box(X_S, BOX_Y0, BOX_Y1)
label(X_S, TITLE_Y, "Structure bank", fs=FS_TITLE, bold=True)
for i, (name, vals) in enumerate(STRUCTURES):
    ax = thumb(X_S, slot_top(i))
    plot_series(ax, vals)
    label(X_S, slot_label_y(i), name)
for yy in (128, 96, 64):  # vertical ellipsis = "more structures" (local level, trend)
    cv.plot([X_S], [yy], "o", ms=6.5, color=COL_INK, zorder=4)

# ---------- Stage 2: noise (solid rounded box) ----------
box(X_N, BOX_Y0, BOX_Y1)
label(X_N, TITLE_Y, "Noise", fs=FS_TITLE, bold=True)
xs = np.linspace(-4.2, 4.2, 400)
gauss = np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi)
nu = 5.0
from math import gamma
tdens = gamma((nu + 1) / 2) / (np.sqrt(nu * np.pi) * gamma(nu / 2)) * (1 + xs ** 2 / nu) ** (-(nu + 1) / 2)
s = 0.45
bimod = 0.5 * (np.exp(-(xs + 1) ** 2 / (2 * s ** 2)) + np.exp(-(xs - 1) ** 2 / (2 * s ** 2))) / (s * np.sqrt(2 * np.pi))
for i, (name, d) in enumerate(zip(NOISES, (gauss, tdens, bimod))):
    ax = thumb(X_N, slot_top(i))
    ax.plot(xs, d, color=COL_INK, lw=LW_INK)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-0.06 * d.max(), 1.12 * d.max())
    label(X_N, slot_label_y(i), name)
label(X_N, slot_top(3) - 48, "+ GARCH", fs=FS_LABEL)
label(X_N, slot_top(3) - 82, "volatility", fs=FS_SMALL)
label(X_N, slot_top(3) - 108, "(optional)", fs=FS_SMALL)

# ---------- Stage 3: break (dashed rounded box) ----------
box(X_B, BOX_Y0, BOX_Y1, dashed=True)
label(X_B, TITLE_Y, "Break", fs=FS_TITLE, bold=True)
wn = np.asarray(WHITE_NOISE, float); wn = (wn - wn.mean()) / wn.std()
m = len(wn) // 2
sch_level = np.r_[np.zeros(m), np.ones(len(wn) - m)]
sch_var = np.r_[0.35 * wn[:m], 1.25 * wn[m:]]
smooth = np.convolve(wn, np.ones(9) / 9, mode="same") * 2.2
sch_dep = np.r_[smooth[:m], wn[m:]]
sch_shape = np.r_[wn[:m], np.sign(wn[m:]) * np.abs(wn[m:]) ** 1.9]
for i, (name, y) in enumerate(zip(BREAKS, (sch_level, sch_var, sch_dep, sch_shape))):
    ax = thumb(X_B, slot_top(i))
    if name == "level":
        xx = np.arange(len(y))
        ax.plot(xx, y, color=COL_INK, lw=LW_INK + 0.4, drawstyle="steps-mid")
        ax.set_xlim(-0.03 * len(y), 1.03 * len(y)); ax.set_ylim(-0.6, 1.6)
    else:
        plot_series(ax, y, lw=LW_INK * 0.85)
    ax.axvline(m, color=COL_RED, lw=1.1, ls=(0, (3, 2)), zorder=0.5)
    label(X_B, slot_label_y(i), name)
label(X_B, 112, "on 50%", fs=FS_LABEL - 1)
label(X_B, 80, "of series", fs=FS_LABEL - 1)

# ---------- arrows between stages (two-line bold labels, ref "sample / kernels") ----------
AY = 500
arrow((X_S + BOX_HALF, AY), (X_N - BOX_HALF - 4, AY))
label((X_S + X_N) / 2, AY + 34, "add", bold=True)
label((X_S + X_N) / 2, AY - 34, "noise", bold=True)
arrow((X_N + BOX_HALF, AY), (X_B - BOX_HALF - 4, AY))
label((X_N + X_B) / 2, AY + 34, "sample", bold=True)
label((X_N + X_B) / 2, AY - 34, "break", bold=True)

# ---------- output column: template (black) ~ sample (blue) ----------
OUT_W, OUT_H = THUMB, THUMB   # iter1: result tiles in bank-tile register (near-square, same size)
X_O = X_B + P_OUT
TILDE_GAP = 70                # ref gap between sample tile and kernel tile ~0.42 tile
TOP_LO = AY + OUT_H / 2       # lower (template) tile centred on the stage arrow row
TOP_UP = TOP_LO + TILDE_GAP + OUT_H
out = np.asarray(OUTPUT, float)
ax_up = thumb(X_O, TOP_UP, OUT_W, OUT_H)
plot_series(ax_up, out, color=COL_BLUE, lw=0.7, pad=0.1)
ax_up.axvline(OUTPUT_N0, color=COL_RED, lw=1.2, ls=(0, (3, 2)))
yl = ax_up.get_ylim()
ax_up.set_ylim(yl[0], yl[1] + 0.22 * (yl[1] - yl[0]))
ax_up.text(OUTPUT_N0 + 8, yl[1] + 0.02 * (yl[1] - yl[0]), r"$n_0$", color=COL_RED,
           fontsize=FS_SMALL, ha="left", va="bottom", zorder=10)
# wreath "~" between stacked panels (ref uses a vertical tilde)
cv.text(X_O, (TOP_UP - OUT_H + TOP_LO) / 2, "~", ha="center", va="center", rotation=90,
        fontsize=54, color=COL_GREY, zorder=10)   # ref: rotated tilde, grey, ~0.3 tile tall
ax_lo = thumb(X_O, TOP_LO, OUT_W, OUT_H)
mu0, mu1 = out[:OUTPUT_N0].mean(), out[OUTPUT_N0:].mean()
tmpl = np.r_[np.full(OUTPUT_N0, mu0), np.full(len(out) - OUTPUT_N0, mu1)]
ax_lo.plot(np.arange(len(out)), tmpl, color=COL_INK, lw=LW_INK + 0.6, drawstyle="steps-mid")
ax_lo.axvline(OUTPUT_N0, color=COL_RED, lw=1.2, ls=(0, (3, 2)))
ax_lo.set_xlim(-0.03 * len(out), 1.03 * len(out))
r = mu0 - mu1
ax_lo.set_ylim(mu1 - 1.2 * r, mu0 + 1.2 * r)
LO_BOTTOM = TOP_LO - OUT_H
label(X_O, LO_BOTTOM - 40, "structure + noise")
label(X_O, LO_BOTTOM - 80, "+", fs=FS_LABEL + 4)
label(X_O, LO_BOTTOM - 120, "level break at $n_0$")
label(X_O, LO_BOTTOM - 168, "label = 1", bold=True)

# ---------- elbow connector: sampled break (level) -> output ----------
LV_Y = slot_top(0) - THUMB / 2
XE = X_O - OUT_W / 2 - 105   # iter2: short final arrow run (~0.75 tile, ref ~0.7 tile)
LO_MID = TOP_LO - OUT_H / 2
X_AR = XE
cv.plot([X_B + THUMB / 2, XE, XE, X_AR + 2], [LV_Y, LV_Y, LO_MID, LO_MID], color=COL_GREY, lw=LW_BOX,
        solid_joinstyle="miter", solid_capstyle="butt", zorder=1.5)
arrow((X_AR, LO_MID), (X_O - OUT_W / 2 - 4, LO_MID))

fig.savefig("figure.png", dpi=DPI, facecolor="white"); fig.savefig("figure.pdf", dpi=DPI, facecolor="white"); fig.savefig("output.png", dpi=DPI, facecolor="white")


# ---------- floor self-check ----------
def floor_check():
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [t for ax in fig.axes for t in ax.texts if t.get_text() and t.get_visible()]
    bbs = [(t.get_text(), t.get_window_extent(rend)) for t in texts]
    over = [f"'{a}' <-> '{b}'" for i, (a, ba) in enumerate(bbs) for b, bb in bbs[i + 1:]
            if ba.overlaps(bb)]
    clip = [a for a, b in bbs if not (fig.bbox.x0 <= b.x0 and b.x1 <= fig.bbox.x1
                                       and fig.bbox.y0 <= b.y0 and b.y1 <= fig.bbox.y1)]
    # text vs thumbnail axes (plotted layers)
    obsc = []
    for a, b in bbs:
        for ax in fig.axes[1:]:
            if b.overlaps(ax.bbox) and a not in ("$n_0$",):
                obsc.append(f"'{a}' over thumbnail")
    lines = [f"text_overlap: {'PASS' if not over else 'FAIL ' + '; '.join(over)}",
             f"text_clipped: {'PASS' if not clip else 'FAIL ' + '; '.join(clip)}",
             f"text_obscured_by_marks: {'PASS' if not obsc else 'FAIL ' + '; '.join(obsc)}",
             f"n_texts_checked: {len(bbs)}",
             f"figure_aspect: {FIG_W / FIG_H:.3f} (ref 1.859)",
             f"xlabel_clip_bottom: N/A (diagram, no axis labels)",
             f"cross_panel_label_bleed: PASS (covered by text_overlap + thumbnail-bbox check)",
             f"thumbnail_ticks: none (matches ref)"]
    return lines


if __name__ == "__main__":
    res = floor_check()
    print("\n".join(res))
    with open("floor_selfcheck_final.txt", "w") as f:
        f.write("\n".join(res) + "\n")
