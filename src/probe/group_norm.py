"""Structure-normalized copies of the most useful combo10gnj columns: (x - mean within fitted structure) / sd within
structure, the structure being genA g_struct (WN/AR/SIN/RW/... from the whole-series fit; no labels involved, so the
statistics use train + test rows). Columns: the top 150 by gain of a fast LightGBM fit on the training rows of fold 0
only would leak into the other folds, so the list is taken from the 3-seed combo10gnj gain ranking is avoided too:
all PAT columns with at least 1% non-missing are normalized (cheap: one pass, columns x groups).

    python probe/group_norm.py      # -> outputs/factory/gnorm.parquet (train + test)
"""
import json

import numpy as np
import pandas as pd

PAT = r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_"
keep = json.load(open("outputs/harvest_keep.json"))
ld = lambda p: pd.read_parquet(p).replace([np.inf, -np.inf], np.nan).astype(np.float32)
parts = []
for part in ("train", "test"):
    N = pd.concat([ld(f"outputs/factory/{f}.parquet") for f in ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost", "genA", "genB", "gnoise", "gjlr")], axis=1)
    N.index = N.index.astype(str)
    N = N.loc[:, N.columns.str.contains(PAT, regex=True)]
    F = pd.concat([ld(f"outputs/feat_{part}.parquet"), ld(f"outputs/harvest_{part}.parquet")[keep]], axis=1)
    F.index = F.index.astype(str)
    parts.append(pd.concat([F, N.reindex(F.index)], axis=1))
X = pd.concat(parts)
X = X.loc[:, ~X.columns.duplicated()]
g = pd.read_parquet("outputs/factory/genA.parquet")["g_struct"]
g.index = g.index.astype(str)
g = g.reindex(X.index).fillna(-1)
cols = [c for c in X.columns if c != "g_struct" and X[c].notna().mean() > .01]
G = X[cols].groupby(g.to_numpy())
Z = ((X[cols] - G.transform("mean")) / (G.transform("std") + 1e-9)).astype(np.float32)
Z.columns = [f"gn_{c}" for c in cols]
Z.to_parquet("outputs/factory/gnorm.parquet")
print(Z.shape, "groups", g.value_counts().to_dict())
