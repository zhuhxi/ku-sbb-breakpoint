"""One knob at a time on menu "L abc30": real-vs-fake AUC (break series only) for each value.

    python probe/tune_abc.py rate      # level-change rate
    python probe/tune_abc.py drift     # size of the boundary drift
    python probe/tune_abc.py small     # upper end of the small group
    python probe/tune_abc.py wnbig     # extra large-jump chance for white-noise series
    python probe/tune_abc.py big       # width of the large group
    python probe/tune_abc.py share     # small / medium / large shares
    python probe/tune_abc.py nswitch   # chance that a break also redraws the noise type
"""
import sys
import time

knob = sys.argv[1]
extra = sys.argv[2:]
sys.argv = [sys.argv[0], "--menu", "L abc30", "--only-break", "1"]
import importlib.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

s = importlib.util.spec_from_file_location("bc", "probe/break_c2st.py")
bc = importlib.util.module_from_spec(s); s.loader.exec_module(bc)
base = bc.cb.BREAKS["L abc30"]
small, mid, big = list(np.round(np.linspace(.01, .49, 76), 3)), lambda c: list(np.round(np.linspace(c - .05, c + .05, 11), 3)), list(np.round(np.linspace(1.8, 2.2, 13), 3))
if knob == "rate":
    menus = {f"rate {q:.0%}": dict(base, switches=dict(base["switches"], level=q)) for q in (.2, .3, .4, .5)}
elif knob == "nswitch":
    base = bc.cb.BREAKS["L abc40"]
    menus = {f"switch {q:.0%}": dict(base, noise_switch=q) for q in (0, .2, .35, .5)}
elif knob == "share":                                      # small / medium / large group shares of the level jumps
    base = bc.cb.BREAKS["L abc40 ns35"]
    grp = lambda a, b, k: list(np.round(np.linspace(a, b, k), 3))
    menus = {f"share {a}/{b}/{c}": dict(base, level=(grp(.01, .49, a) + grp(.8, .9, b) + grp(1.8, 2.2, c),))
             for a, b, c in [(76, 11, 13), (65, 15, 20), (55, 20, 25), (45, 25, 30)]}
elif knob == "big":                                        # width of the large group (shares stay 76/11/13)
    base = bc.cb.BREAKS["L abc40 ns35"]
    grp = lambda a, b, k: list(np.round(np.linspace(a, b, k), 3))
    menus = {f"big {a}-{b}": dict(base, level=(grp(.01, .49, 76) + grp(.8, .9, 11) + grp(a, b, 13),))
             for a, b in [(1.8, 2.2), (1.5, 2.5), (1.2, 2.8), (1.0, 3.0)]}
elif knob == "wnbig":                                      # extra chance that a white-noise level jump is large
    base = bc.cb.BREAKS["L abc40 ns35"]
    menus = {f"wn_big {q:.0%}": dict(base, wn_big=q) for q in (0, .05, .1, .15)}
elif knob == "small":                                      # upper end of the small group (same random draws, only sizes change)
    base = bc.cb.BREAKS["L abc40 ns35"]
    grp = lambda a, b, k: list(np.round(np.linspace(a, b, k), 3))
    menus = {f"small 0-{h}": dict(base, level=(grp(.01, h, 76) + grp(.8, .9, 11) + grp(1.8, 2.2, 13),)) for h in [float(v) for v in (extra or [.5, .4, .3, .2])]}
elif knob == "drift":                                      # size of the drift that starts at the boundary
    base = bc.cb.BREAKS["L abc40 ns35"]
    menus = {f"drift {a}-{b}": dict(base, drift_size=(a, b)) for a, b in [(.5, 2), (.25, 1), (.1, .5), (0, 0)]}
elif knob == "mid":
    menus = {f"mid {c}": dict(base, level=(small + mid(c) + big,)) for c in (.7, .85, 1.0)}

t0 = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = set(np.random.default_rng(7).choice(lab.index[lab == 1], 1500, False))
real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
for name, M in menus.items():
    fake = Parallel(n_jobs=30)(delayed(bc.cb.gen)(M, 5 * 10 ** 6 + 10 ** 5 + i, True, True) for i in range(1500))
    a, tt = bc.c2st(real, fake)
    print(f"{name:10s} AUC {a:.3f}   tell-tale {tt[:2]}")
print(f"total {time.time() - t0:.0f}s")
