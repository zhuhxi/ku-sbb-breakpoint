"""Where do generated level changes differ from real ones? Quantiles of level-related before/after statistics,
real break series vs generated (menu), stationary pre only, split by pre-boundary structure.

    python probe/level_cmp.py MENU
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

name = sys.argv[1]
sys.argv = [sys.argv[0], "--n", "1"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
c2 = load("c2", "probe/break_c2st.py")
bm = load("bm", "probe/break_measure.py")


def feats(v, n0):
    f = c2.feats(v, n0)
    f["lvl_noise"], f["spr_noise"], f["dep_ar1"] = bm.measure(v, n0)
    return f


lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = set(np.random.default_rng(3).choice(lab.index[lab == 1], 2500, False))
real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
fake = Parallel(n_jobs=30)(delayed(c2.cb.gen)(c2.cb.BREAKS[name], 7 * 10 ** 6 + s, True, True) for s in range(2500))
R = pd.DataFrame(Parallel(n_jobs=30)(delayed(feats)(v, n0) for v, n0 in real))
G = pd.DataFrame(Parallel(n_jobs=30)(delayed(feats)(v, n0) for v, n0 in fake))
R, G = R[R.cls.isin(["WN", "AR1", "AR2"])], G[G.cls.isin(["WN", "AR1", "AR2"])]
print("pre structure share  real", R.cls.value_counts(normalize=True).round(3).to_dict(), " gen", G.cls.value_counts(normalize=True).round(3).to_dict())
qs = [.5, .75, .9, .95, .99]
for c in ["level", "lvl_noise", "robust level (median)", "gradual: level late vs early after", "local: first 30 after vs last 30 before", "spread", "spr_noise", "dep"]:
    a, b = R[c].abs() if c in ("lvl_noise",) else R[c], G[c].abs() if c in ("lvl_noise",) else G[c]
    if c == "spr_noise":
        a, b = np.abs(np.log(R[c])), np.abs(np.log(G[c]))
    print(f"{c:42s} real {np.round(np.quantile(a, qs), 2)}  gen {np.round(np.quantile(b, qs), 2)}")
for k in ["WN", "AR1", "AR2"]:
    a, b = R.loc[R.cls == k, "lvl_noise"].abs(), G.loc[G.cls == k, "lvl_noise"].abs()
    print(f"  |level| noise units, pre={k:3s} real {np.round(np.quantile(a, qs), 2)}  gen {np.round(np.quantile(b, qs), 2)}")
