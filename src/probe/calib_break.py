"""Simulation calibration of the break menu. Stationary pre-boundary structures (WN / AR1 / AR2 from the calibrated
no-break menu, with the trend layer) get one of the three official changes at the boundary:
  level  : add d * sd(series) after the boundary
  spread : multiply innovations after the boundary by k
  dep    : AR(1) coefficient after the boundary = before + delta (clipped to +-0.95)
The same before-vs-after measurements (probe/break_size.py) are taken on real and simulated break series whose
pre-boundary part is classified WN / AR1 / AR2, and their distributions compared (histogram L1, lower = closer).

    python probe/calib_break.py MENU [MENU ...]
"""
import copy
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

menus = sys.argv[1:]
sys.argv = [sys.argv[0], "x"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
cp = load("cp", "probe/calib_params.py")
bs = load("bs", "probe/break_size.py")
cc = cp.cc
U = lambda r, *a: r.choice([-1, 1]) * (r.uniform(*a) if len(a) == 2 else r.choice(a[0]))   # (lo, hi) uniform, ([..],) discrete
BREAKS = {   # mechanism shares; level d range (sd units); spread: log k range; dep: delta range
    "thirds": dict(p=dict(level=1 / 3, spread=1 / 3, dep=1 / 3), level=(0, 2.5), spread=(0, .75), dep=(0, .6)),
    "4-4-2": dict(p=dict(level=.4, spread=.4, dep=.2), level=(0, 2.5), spread=(0, .75), dep=(0, .6)),
    "thirds-wide": dict(p=dict(level=1 / 3, spread=1 / 3, dep=1 / 3), level=(0, 3), spread=(0, 1), dep=(0, .8)),
    "thirds-min": dict(p=dict(level=1 / 3, spread=1 / 3, dep=1 / 3), level=(.5, 2.5), spread=(.2, .75), dep=(.2, .6)),
    "4-4-2 small": dict(p=dict(level=.4, spread=.4, dep=.2), level=(0, 2), spread=(0, .6), dep=(0, .4)),
    "45-45-10 small": dict(p=dict(level=.45, spread=.45, dep=.1), level=(0, 2), spread=(0, .6), dep=(0, .5)),
    "4-4-2 smaller": dict(p=dict(level=.4, spread=.4, dep=.2), level=(0, 1.6), spread=(0, .5), dep=(0, .35)),
    # two strengths (weak / strong), suggested by the gaps in the real size histograms
    "two-strength": dict(p=dict(level=.4, spread=.4, dep=.2), level=([.4, 2.0],), spread=([.3, np.log(2)],), dep=([.2, .45],)),
    "two-strength b": dict(p=dict(level=.4, spread=.4, dep=.2), level=([.3, 2.0],), spread=([.25, np.log(2)],), dep=([.15, .45],)),
    "two-strength c": dict(p=dict(level=.4, spread=.4, dep=.2), level=([.5, 2.0],), spread=([.35, np.log(2)],), dep=([.25, .5],)),
    "weak-mostly": dict(p=dict(level=.4, spread=.4, dep=.2), level=([.4, .4, .4, 2.0],), spread=([.3, .3, np.log(2)],), dep=([.2, .2, .45],)),
    "weak-mostly unif": dict(p=dict(level=.4, spread=.4, dep=.2), level=([.2, .3, .4, .5, .6, .2, .3, .4, .5, .6, 1.8, 1.9, 2.0, 2.1, 2.2],),
                             spread=([.2, .25, .3, .35, .4, .2, .25, .3, .35, .4, .65, .7, .75],), dep=([.15, .2, .25, .3, .15, .2, .25, .3, .4, .5],)),
    "weak-mostly unif2": dict(p=dict(level=.4, spread=.4, dep=.2), level=([.2, .3, .4, .5, .6, .2, .3, .4, .5, .6, 1.8, 1.9, 2.0, 2.1, 2.2],),
                              spread=([.15, .2, .25, .3, .35, .15, .2, .25, .3, .35, .15, .2, .25, .3, .35, .65, .7, .75],), dep=([.15, .2, .25, .3, .15, .2, .25, .3, .45],)),
    "weak-mostly unif3": dict(p=dict(level=.35, spread=.45, dep=.2), level=([.2, .3, .4, .5, .6, .2, .3, .4, .5, .6, 1.8, 1.9, 2.0, 2.1, 2.2],),
                              spread=([.15, .2, .25, .3, .35, .15, .2, .25, .3, .35, .65, .7, .75],), dep=([.15, .2, .25, .3, .15, .2, .25, .3, .45],)),
    # read off the per-series measurements (probe/break_measure.py): level in noise-sd units, strong +-2;
    # spread strong x2 / x0.5, weak x1.25-1.6; strong = 1/3 of each type
    "measured": dict(p=dict(level=.4, spread=.4, dep=.2), level_unit="noise",
                     level=(list(np.round(np.linspace(.25, 1.25, 6), 2)) * 2 + [2.0] * 6,),
                     spread=(list(np.round(np.log(np.linspace(1.25, 1.6, 6)), 3)) * 2 + [np.log(2)] * 6,),
                     dep=([.1, .14, .18, .22, .25] * 9 + [.5] * 5,)),
    "measured2": dict(p=dict(level=.4, spread=.4, dep=.2), level_unit="noise",
                      level=(list(np.round(np.linspace(.2, 1.0, 5), 2)) * 5 + [2.0] * 5,),
                      spread=(list(np.round(np.log(np.linspace(1.1, 1.45, 5)), 3)) * 5 + [np.log(2)] * 5,),
                      dep=([.1, .14, .18, .22, .25] * 9 + [.5] * 5,)),
    "measured2 keepvar": None,
}
BREAKS["measured2 keepvar"] = dict(BREAKS["measured2"], dep_keepvar=True)
LV3 = (list(np.round(np.linspace(.2, 1.25, 6), 2)) * 3 + [1.8, 1.9, 2.0, 2.1, 2.2, 2.0],)
BREAKS["both10"] = dict(BREAKS["measured2"], p=dict(level=.35, spread=.35, dep=.2, both=.1))
BREAKS["both20"] = dict(BREAKS["measured2"], p=dict(level=.3, spread=.3, dep=.2, both=.2))
BREAKS["both33"] = dict(BREAKS["measured2"], p=dict(level=.25, spread=.25, dep=.17, both=.33))
LV4 = (list(np.round(np.linspace(.2, 1.0, 5), 2)) * 5 + list(np.round(np.linspace(1.75, 3.5, 8), 2)),)
SP4 = (list(np.round(np.log(np.linspace(1.1, 1.45, 5)), 3)) * 5 + [np.log(2)] * 5,)
BREAKS["m4"] = dict(BREAKS["measured2"], p=dict(level=.42, spread=.3, dep=.2, both=.08), level=LV4, spread=SP4)
BREAKS["m4b"] = dict(BREAKS["measured2"], p=dict(level=.45, spread=.3, dep=.15, both=.1), level=LV4, spread=SP4,
                     dep=([.05, .1, .15, .2] * 9 + [.5] * 4,))
LV5 = (list(np.round(np.linspace(.1, .9, 5), 2)) * 6 + [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5, 3.2],)
SP5 = (list(np.round(np.log(np.linspace(1.05, 1.4, 5)), 3)) * 6 + [np.log(2)] * 6,)
DP5 = ([.05, .1, .15, .2] * 9 + [.5] * 4,)
BREAKS["m5"] = dict(BREAKS["measured2"], p=dict(level=.42, spread=.33, dep=.17, both=.08), level=LV5, spread=SP5, dep=DP5)
BREAKS["m5b"] = dict(BREAKS["measured2"], p=dict(level=.4, spread=.3, dep=.2, both=.1), level=LV5, spread=SP5, dep=DP5)
BREAKS["A switches"] = dict(switches=dict(level=.45, spread=.35, dep=.2), level=LV5, spread=SP5, dep=DP5)
BREAKS["B redraw"] = dict(redraw=True)
for _q in (3, 5, 8):
    BREAKS[f"A drift{_q}"] = dict(BREAKS["A switches"], drift_p=_q / 100)
_wk = list(np.round(np.linspace(0.05, 0.3, 10), 3))
BREAKS["L tuned"] = dict(BREAKS["A drift8"], switches=dict(level=.3, spread=.35, dep=.2),
                         level=(_wk * 16 + list(np.round(np.linspace(1.8, 2.2, 5), 2)) * 8,))   # strong 1.8-2.2 on 20%
# read off ABC (probe/abc_level.py): three level groups, small 0-.5 (76%), medium ~.85 (11%), large 1.8-2.2 (13%)
BREAKS["L abc"] = dict(BREAKS["L tuned"], switches=dict(level=.75, spread=.35, dep=.2),
                       level=(list(np.round(np.linspace(.01, .49, 76), 3)) + list(np.round(np.linspace(.8, .9, 11), 3))
                              + list(np.round(np.linspace(1.8, 2.2, 13), 3)),))
BREAKS["L abc30"] = dict(BREAKS["L abc"], switches=dict(level=.3, spread=.35, dep=.2))   # same three groups, old 30% rate
BREAKS["L abc40"] = dict(BREAKS["L abc"], switches=dict(level=.4, spread=.35, dep=.2))   # tuned rate (probe/tune_abc.py)
BREAKS["L abc40 ns35"] = dict(BREAKS["L abc40"], noise_switch=.35)   # best so far: break C2ST 0.607 (probe/tune_abc.py nswitch)
BREAKS["L abc40 ns35 shell"] = dict(BREAKS["L abc40 ns35"], shell=True)   # + outer base / size, label-independent
BREAKS["L abc40 ns35 shell s6"] = dict(BREAKS["L abc40 ns35 shell"], struct6=True)   # + SIN / RW / local level
# re-measured on white-noise-like real series (probe/break_sizes_real.py): level small .3-.7, medium ~1.1, large ~2;
# spread weak x1.3-1.6, strong x2
BREAKS["sz1"] = dict(BREAKS["L abc40 ns35 shell s6"],
                     level=(list(np.round(np.linspace(.3, .7, 50), 3)) + list(np.round(np.linspace(1.0, 1.2, 20), 3))
                            + list(np.round(np.linspace(1.8, 2.2, 30), 3)),),
                     spread=(list(np.round(np.log(np.linspace(1.3, 1.6, 5)), 3)) * 4 + [np.log(2)] * 8,))
# probe/dep_change.py: real dependence breaks are rare, and ~40% of real label-1 series show no visible change
BREAKS["sz2"] = dict(BREAKS["sz1"], switches=dict(level=.4, spread=.35, dep=.05), null_p=.35)
BREAKS["sz3"] = dict(BREAKS["sz2"], spread=(list(np.round(np.log(np.linspace(1.2, 1.45, 5)), 3)) * 4 + [np.log(2)] * 8,))   # weak spread x1.2-1.45
for _c in (8, 6, 4):                                        # spread (log k) and level jumps scaled by 0.8 / 0.6 / 0.4
    BREAKS[f"s6 shrink{_c}"] = dict(BREAKS["L abc40 ns35 shell s6"], shrink=_c / 10)
# probe/fp6b.py: fewer level jumps, part of them through the intercept; spread change scaled by .6 (probe/break_by_struct_fake.py)
BREAKS["c1"] = dict(BREAKS["L abc40 ns35 shell s6"], shrink=.6, lshrink=1, switches=dict(level=.2, spread=.35, dep=.2), icp_p=.25)
BREAKS["c2"] = dict(BREAKS["L abc40 ns35 shell s6"], shrink=.6, lshrink=1, switches=dict(level=.25, spread=.35, dep=.2), icp_p=.4)
for _m in ("drop", "slow", "arma"):                          # local-level variants (probe/fp6e.py: ours steps before the boundary)
    BREAKS[f"s6 LL {_m}"] = dict(BREAKS["L abc40 ns35 shell s6"], ll_mode=_m)
BREAKS["s6 sout"] = dict(BREAKS["L abc40 ns35 shell s6"], spread_out=True)     # spread change on the output
BREAKS["s6 sout shrink6"] = dict(BREAKS["s6 sout"], shrink=.6, lshrink=1)
BREAKS["ok1"] = dict(BREAKS["s6 sout shrink6"], one_kind=dict(spread=.45, level=.25, dep=.15, shape=.15))   # one change per break
BREAKS["ok2"] = dict(BREAKS["ok1"], shrink=1)                                      # spread not shrunk
BREAKS["ok3"] = dict(BREAKS["ok1"], shrink=.8)
BREAKS["ok4"] = dict(BREAKS["ok1"], shrink=1, one_kind=dict(spread=.45, level=.35, dep=.1, shape=.1))
BREAKS["ok5"] = dict(BREAKS["ok4"], level=(.3, 2.0))                               # ekhi level jump: +-U(.3, 2) noise sd
# probe/invisible.py: break sizes continuous with most mass near 0 -> N(0, sd) (synfin); sd matched to ok4's mean |size|
BREAKS["ok6"] = dict(BREAKS["ok4"], cont=dict(spread=.35, level=.69, dep=.2))
BREAKS["ok7"] = dict(BREAKS["ok4"], cont=dict(spread=.45, level=.9, dep=.26))
BREAKS["ok8"] = dict(BREAKS["ok4"], decoy=dict(p=.05, size=(.3, 1.2), len=(8, 16)))   # LESSONS 11-04: short bump after n0 in no-break
BREAKS["ok9"] = dict(BREAKS["ok4"], decoy=dict(p=.08, size=(.5, 2.0), len=(8, 16)))
BREAKS["ok11"] = dict(BREAKS["ok4"], decoy=dict(p=.08, size=(.5, 2.0), len=(10, 22), smooth=True))
BREAKS["ok12"] = dict(BREAKS["ok4"], decoy=dict(p=.10, size=(.4, 1.6), len=(10, 22), smooth=True))
BREAKS["ok13"] = dict(BREAKS["ok12"], vburst=dict(p=.06, k=(1.5, 2.5), len=(10, 22)))
BREAKS["ok14"] = dict(BREAKS["ok12"], vburst=dict(p=.06, k=(1.5, 2.8), len=(8, 18), smooth=True))
BREAKS["ok15"] = dict(BREAKS["ok14"], ar_norm=True)
BREAKS["ok16"] = dict(BREAKS["ok15"], struct_p=[.42, .17, .10, .15, .09, .07], sin_scale=.86)   # gen_compare: RW too rare, SIN too big/common
BREAKS["ok17"] = dict(BREAKS["ok16"], struct_p=[.43, .17, .10, .15, .09, .06])
BREAKS["ok18"] = dict(BREAKS["ok16"], t_df=(3.5, 10))                       # heavier tails (real kurt / spikes higher)
BREAKS["ok19"] = dict(BREAKS["ok16"], ar1=(-.5, .7))                         # fewer strongly negative AR1
BREAKS["ok20"] = dict(BREAKS["ok16"], t_df=(3.5, 10), ar1=(-.5, .7))
BREAKS["ok21"] = dict(BREAKS["ok19"], t_df=(4.5, 10))
BREAKS["ok22"] = dict(BREAKS["ok19"], t_df=(4, 12))
BREAKS["ok23"] = dict(BREAKS["ok21"], lshrink=1.25)                        # bigger level jumps (real dm tail)
BREAKS["ok24"] = dict(BREAKS["ok21"], one_kind=dict(spread=.40, level=.40, dep=.1, shape=.1))
BREAKS["ok25"] = dict(BREAKS["ok21"], lshrink=1.25, one_kind=dict(spread=.40, level=.40, dep=.1, shape=.1))
BREAKS["ok26"] = dict(BREAKS["ok25"], one_kind=dict(spread=.40, level=.42, dep=.06, shape=.12))
X1 = dict(BREAKS["ok25"], sin_amp=(.2, 2.5))
BREAKS["X2"] = dict(BREAKS["ok25"], sin_log=(.15, 3))
BREAKS["X3"] = dict(BREAKS["ok25"], lvl_series=True)
BREAKS["X4"] = dict(BREAKS["ok25"], level=([.05, .1, .15, .2, .25, .3, .35, .4, .45] * 4 + [.7] * 6 + [1.0] * 6 + [2.0] * 4,))
BREAKS["X5"] = dict(BREAKS["ok25"], dep=([.05, .1] * 4 + [.15, .25, .3] * 4 + [.5] * 2,))
BREAKS["X6"] = dict(BREAKS["ok25"], ar1=(-.8, .9))
BREAKS["X8"] = dict(BREAKS["ok25"], ll_sd=(.03, .15))
BREAKS["X9"] = dict(BREAKS["ok25"], garch_p=.4)
BREAKS["X10"] = dict(BREAKS["ok25"], garch_p=.1)
BREAKS["X1"] = X1
BREAKS["X7"] = dict(BREAKS["ok25"], ar1=(-.5, .8))                        # ekhi reference: AR1 phi U(-.5, .8)
BREAKS["ok27"] = dict(X1, ar1=(-.5, .8))
BREAKS["ok10"] = dict(BREAKS["ok4"], decoy=dict(p=.06, size=(1.0, 3.0), len=(8, 16)))
BREAKS["L abc40 ns35 kb"] = dict(BREAKS["L abc40 ns35"], keep_bimodal=True)   # two-level noise is never switched
for _q in (35, 70, 100):                                     # redraw the noise parameters instead of switching the family
    BREAKS[f"L abc40 rd{_q}"] = dict(BREAKS["L abc40 ns35"], noise_switch=0, noise_redraw=_q / 100)
BREAKS["m3 series"] = dict(BREAKS["measured2"], level=LV3, level_unit="series")
BREAKS["m3 noise"] = dict(BREAKS["measured2"], level=LV3, level_unit="noise")


def ar2_var(p1, p2):
    """Stationary variance of an AR(2) with unit innovations."""
    return (1 - p2) / ((1 + p2) * ((1 - p2) ** 2 - p1 ** 2))


def ar_path(phis, e):
    """AR(2) recursion with time-varying coefficients phis[t] = (p1, p2)."""
    x = np.zeros(len(e))
    for t in range(len(e)):
        x[t] = e[t] + (phis[t, 0] * x[t - 1] if t > 0 else 0) + (phis[t, 1] * x[t - 2] if t > 1 else 0)
    return x


def gen(B, seed, brk=True, series=False):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    n0 = int(r.uniform(0.3, 0.7) * n)
    t = np.arange(n)
    null = bool(B and B.get("null_p")) and brk and r.random() < B["null_p"]   # break with no change at all
    if null:
        brk = False
    dfr = B.get("t_df", (5, 10)) if B else (5, 10)
    gp = B.get("garch_p", .25) if B else .25
    bimodal = copy.deepcopy(r).random() < 0.05               # cc.noise picks two-level (bimodal) noise when its first draw < .05
    e = cc.noise(r, n, dfr, gp)
    nsw = False
    ok = None
    sp_size = dp_size = 0.0                                  # applied spread |log k| and dependence |delta| (for "info")
    if brk and B.get("one_kind"):                            # probe/one_kind.py: a break changes exactly one thing
        ok = r.choice(list(B["one_kind"]), p=list(B["one_kind"].values()))
    if brk and r.random() < B.get("noise_redraw", 0):         # same noise family after the boundary, its parameters redrawn
        u0 = copy.deepcopy(r)
        u = u0.random()
        if u < .05:
            e[n0:] = r.choice([-1, 1], n - n0) * r.uniform(1.0, 2.5) + r.standard_normal(n - n0)
        elif u < .525:
            df = r.uniform(5, 10)
            e[n0:] = r.standard_t(df, n - n0) / np.sqrt(df / (df - 2))
    if brk and (ok == "shape" if ok else r.random() < B.get("noise_switch", 0)) and not (B.get("keep_bimodal") and bimodal):      # the break also redraws the noise type after the boundary
        e[n0:] = cc.noise(r, n - n0, dfr, gp)                          # unit-variance noise, no matching of the sample sd
        nsw = True
    if B and B.get("struct6"):                               # all six no-break structures (probe/calib_params.py shares)
        k = r.choice(["WN", "AR1", "AR2", "SIN", "RW", "LL"], p=B.get("struct_p", [.42, .17, .10, .17, .07, .07]))
    else:
        k = r.choice(["WN", "AR1", "AR2"], p=[.42 / .69, .17 / .69, .10 / .69])
    p1, p2 = 0.0, 0.0
    if k == "AR1":
        p1 = r.uniform(*(B.get("ar1", (-.7, .7)) if B else (-.7, .7)))
    elif k == "AR2":
        while True:
            p1, p2 = r.uniform(-.9, .9), r.uniform(-.5, .5)
            if p2 + p1 < 1 and p2 - p1 < 1:
                break
    phis = np.tile([p1, p2], (n, 1))
    pre_phi = (p1, p2)                                       # for "info" (oracle tests)
    if B and B.get("ar_norm") and k in ("AR1", "AR2"):       # probe/scale_cause.py: real AR series have ~unit marginal size
        e = e / np.sqrt(ar2_var(p1, p2))
    m = r.choice(list(B["p"]), p=list(B["p"].values())) if brk and "p" in B else None
    if brk and B.get("switches"):                           # A: each change on/off independently, at least one on
        while not ok:
            on = {k: r.random() < q for k, q in B["switches"].items()}
            if any(on.values()):
                break
        if ok:
            on = {k2: k2 == ok for k2 in B["switches"]}
        m = "switch"
        if on["spread"]:
            kout = np.exp(r.normal(0, B["cont"]["spread"]) if B.get("cont") else B.get("shrink", 1) * U(r, *B["spread"]))
            sp_size = abs(np.log(kout))
            if B.get("spread_out") and k in ("WN", "AR1", "AR2"):   # probe/fp7.py: scale the output, not the innovations
                pass
            else:
                e[n0:] *= kout
        if on["dep"]:
            dd = (lambda: r.normal(0, B["cont"]["dep"])) if B.get("cont") else (lambda: U(r, *B["dep"]))
            q1 = np.clip(p1 + dd(), -.95, .95)
            while not (p2 + q1 < .98 and p2 - q1 < .98):
                q1 = np.clip(p1 + dd(), -.95, .95)
                p2 *= 0.9
            phis[n0:] = [q1, p2]
            dp_size = abs(q1 - p1)
    if brk and B.get("redraw"):                             # B: redraw every parameter after the boundary
        m = "redraw"
        sd0, sd1 = 2 ** r.uniform(-1, 1), 2 ** r.uniform(-1, 1)
        e[:n0] *= sd0
        e[n0:] *= sd1
        if k == "AR1":
            phis[n0:, 0] = r.uniform(-.7, .7)
        elif k == "AR2":
            while True:
                q1, q2 = r.uniform(-.9, .9), r.uniform(-.5, .5)
                if q2 + q1 < 1 and q2 - q1 < 1:
                    break
            phis[n0:] = [q1, q2]
    if m == "both":                                          # level and spread change together
        e[n0:] *= np.exp(U(r, *B["spread"]))
    if m == "spread":
        e[n0:] *= np.exp(U(r, *B["spread"]))
    if m == "dep":
        p20 = p2
        q1 = np.clip(p1 + U(r, *B["dep"]), -.95, .95)
        while not (p2 + q1 < .98 and p2 - q1 < .98):           # keep the post-boundary AR(2) stationary
            q1 = np.clip(p1 + U(r, *B["dep"]), -.95, .95)
            p2 *= 0.9
        phis[n0:] = [q1, p2]
        if B.get("dep_keepvar"):                               # scale innovations so the series variance stays the same
            e[n0:] *= np.sqrt(ar2_var(p1, p20) / ar2_var(q1, p2))
    x = ar_path(phis, e)
    if m == "switch" and on["spread"] and B.get("spread_out") and k in ("WN", "AR1", "AR2"):
        x[n0:] *= kout
    if k == "SIN":                                            # sine + noise; the break acts on the noise part
        amp = np.exp(r.uniform(*np.log(B["sin_log"]))) if B.get("sin_log") else r.uniform(*B.get("sin_amp", (0.5, 3)))
        x = (x + amp * np.sin(2 * np.pi * t / r.integers(12, 33) + r.uniform(0, 6.3))) * B.get("sin_scale", 1)
    elif k == "RW":                                           # random walk; the break acts on the steps
        x = np.cumsum(x)
    elif k == "LL":                                           # local level: random-walk level + noise
        lm = B.get("ll_mode") if B else None
        if lm == "arma":                                      # ARMA(1,1) near a unit root instead: y = phi y[-1] + x + theta x[-1]
            ph, th = r.uniform(.85, .97), r.uniform(-.6, -.3)
            xm = np.r_[0, x[:-1]]
            x = ar_path(np.tile([ph, 0], (n, 1)), x + th * xm)
        elif lm != "drop":
            x = x + np.cumsum(cc.noise(r, n, dfr, gp) * (r.uniform(.005, .02) if lm == "slow" else r.uniform(*B.get("ll_sd", (.02, .1)))))
    if m == "redraw":
        x[:n0] += r.uniform(-2, 2)
        x[n0:] += r.uniform(-2, 2)
    lv = 0.0                                                # applied level jump, noise-sd units
    if m == "switch" and on["level"]:
        lv = r.normal(0, B["cont"]["level"]) if B.get("cont") else B.get("lshrink", B.get("shrink", 1)) * U(r, *B["level"])
        if k == "WN" and r.random() < B.get("wn_big", 0):   # white-noise series get a large jump more often
            lv = np.sign(lv) * r.uniform(1.8, 2.2)
        if B.get("icp_p") and r.random() < B["icp_p"]:      # through the AR intercept: the mean climbs over ~1/(1-phi) points
            st = np.zeros(n); st[n0:] = lv * e[:n0].std()
            x += ar_path(phis, st) if k in ("WN", "AR1", "AR2") else st
        else:
            x[n0:] += lv * (x[:n0].std() if B.get("lvl_series") else e[:n0].std())
    if m in ("level", "both"):
        x[n0:] += U(r, *B["level"]) * (e[:n0].std() if B.get("level_unit") == "noise" else x[:n0].std())
    if B and B.get("decoy") and not brk and r.random() < B["decoy"]["p"]:   # decoy: level bump right after n0 that goes away
        dk = B["decoy"]
        Ld = int(r.integers(*dk["len"]))
        w = np.sin(np.pi * (np.arange(Ld) + .5) / Ld) if dk.get("smooth") else np.ones(Ld)   # smooth: half-sine bump
        x[n0:n0 + Ld] += (U(r, *dk["size"]) * x[:n0].std() * w)[:len(x) - n0]
    if B and B.get("vburst") and not brk and r.random() < B["vburst"]["p"]:   # decoy: short spread burst after n0
        vb = B["vburst"]
        Lv = int(r.integers(*vb["len"]))
        mu = x[:n0].mean()
        kk = r.uniform(*vb["k"])
        env = 1 + (kk - 1) * np.sin(np.pi * (np.arange(Lv) + .5) / Lv) if vb.get("smooth") else np.full(Lv, kk)
        seg = x[n0:n0 + Lv]
        x[n0:n0 + Lv] = mu + (seg - mu) * env[:len(seg)]
    trend = drift = False
    base, size = 0.0, 1.0
    if r.random() < .08 and k in ("WN", "AR1", "AR2"):
        trend = True
        x = x + r.choice([-1, 1]) * r.uniform(.5, 2.5) * x.std() * t / n
    if B and r.random() < B.get("drift_p", 0):              # distractor: slow drift that starts at the boundary
        drift = True
        x[n0:] += r.choice([-1, 1]) * r.uniform(*B.get("drift_size", (.5, 2))) * e[:n0].std() * (t[n0:] - n0) / (n - n0)
    if B and B.get("shell"):                                 # outer shell (probe/base_check*.py): base U(-2,2), size 2**U(-1,1)
        base, size = r.uniform(-2, 2), 2 ** r.uniform(-1, 1)
        x = base + size * x
    if series == "info":                                     # for inspection: what the generator did
        sw = {k2: bool(v2) for k2, v2 in on.items()} if m == "switch" else {}
        return x, n0, dict(structure=k, mech=m, level=abs(lv), spread=sp_size, dep=dp_size, kind=ok, base=base, size=size, pre_phi=pre_phi, noise_switch=nsw, trend=trend, drift=drift, **{f"on_{k2}": v2 for k2, v2 in sw.items()})
    if series == "truth":                                    # for ABC: also return the true level jump
        return x, n0, abs(lv)
    return (x, n0) if series else bs.one(x, n0)


def hist_l1(a, b, lo, hi):
    bins = np.linspace(lo, hi, 41)
    ha = np.histogram(np.clip(a, lo, hi), bins)[0] / len(a)
    hb = np.histogram(np.clip(b, lo, hi), bins)[0] / len(b)
    return np.abs(ha - hb).sum()


if __name__ == "__main__":
    t0 = time.time()
    D = pd.read_parquet("outputs/probe/break_size.parquet")
    R1, R0 = D[D.y == 1], D[D.y == 0]
    rng = dict(level=(-3, 3), spread=(-1.2, 1.2), dep=(-.8, .8))
    S0 = pd.DataFrame(Parallel(n_jobs=30)(delayed(gen)(None, s, False) for s in range(3000)))
    S0 = S0[S0.cls.isin(["WN", "AR1", "AR2"])]
    print("no-break check (sim vs real, = noise floor):", {c: round(hist_l1(S0[c], R0[c], *rng[c]), 3) for c in rng})
    print("real break  |change| quantiles 25/50/75/90:", {c: np.round(R1[c].abs().quantile([.25, .5, .75, .9]).to_numpy(), 2).tolist() for c in rng})
    for name in menus:
        S1 = pd.DataFrame(Parallel(n_jobs=30)(delayed(gen)(BREAKS[name], 10 ** 6 + s) for s in range(3000)))
        S1 = S1[S1.cls.isin(["WN", "AR1", "AR2"])]
        l1 = {c: round(hist_l1(S1[c], R1[c], *rng[c]), 3) for c in rng}
        print(f"{name:12s} L1 {l1} sum {sum(l1.values()):.3f}")
        print(f"{'':12s} |change| q25/50/75/90:", {c: np.round(S1[c].abs().quantile([.25, .5, .75, .9]).to_numpy(), 2).tolist() for c in rng})
    print(f"total {time.time() - t0:.1f}s")
