"""Noise switch vs noise-parameter redraw at the boundary: share of bimodal-before series still bimodal after, overall
type-switch rate, and break real-vs-fake AUC, against real break series (77%, 0.345).

    python probe/redraw_check.py MENU [MENU ...]
"""
import sys

menus = sys.argv[1:]
sys.argv = [sys.argv[0], "--menu", menus[0], "--only-break", "1"]
import importlib.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
bc = load("bc", "probe/break_c2st.py")
sys.argv = [sys.argv[0], "x"]
ns = load("ns", "probe/noise_shape.py")
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = set(np.random.default_rng(7).choice(lab.index[lab == 1], 1500, False))
real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
for m in menus:
    L = Parallel(n_jobs=30)(delayed(bc.cb.gen)(bc.cb.BREAKS[m], 5 * 10 ** 6 + 10 ** 5 + i, True, True) for i in range(1500))
    D = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(ns.shape)(v, n0) for v, n0 in L))
    b = D[D.type_before == "bimodal"]
    print(f"{m:16s} bimodal stays {(b.type_after == 'bimodal').mean() * 100:3.0f}% (real 77)  any switch {(D.type_before != D.type_after).mean():.3f} (real .345)"
          f"  real-vs-fake AUC {bc.c2st(real, L)[0]:.3f}", flush=True)
