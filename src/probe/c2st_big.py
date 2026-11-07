"""Real vs generated with the competition features (combo10 columns, outputs/aug/X_sim.parquet from probe/sim_series.py).
Only series whose fitted structure (probe/gen_feats.py, g_struct) is WN / AR1 / AR2, the structures the generator makes.
Per label: LightGBM 5-fold AUC (0.5 = indistinguishable), top columns by gain and their single-column AUC.
Label 0 is the reference (the no-break generator is close to real there).

    python probe/c2st_big.py [--nomb]   # --nomb: drop the magic B (mb_*, value-format) columns
"""
import importlib.util
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")
t = time.time()
nomb = "--nomb" in sys.argv
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("gf", "probe/gen_feats.py")
gf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gf)
XF = pd.read_parquet("outputs/aug/X_sim.parquet")
meta = pd.read_parquet("outputs/aug/meta_sim.parquet").reindex(XF.index)
d = pd.read_parquet("outputs/aug/series_sim.parquet").sort_values(["id", "time"])
items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in d.groupby("id", sort=False)]
st = Parallel(n_jobs=-1, batch_size=16)(delayed(lambda v, n0: gf.gen_features(v, n0)["g_struct"])(v, n0) for _, v, n0 in items)
sF = pd.Series(st, index=[k for k, _, _ in items]).reindex(XF.index)
print(f"fake structures fitted {time.time() - t:.0f}s", flush=True)
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
ld = lambda p: pd.read_parquet(p).reindex(lab.index)
import json
N = pd.concat([ld(f"outputs/factory/{f}.parquet") for f in ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")], axis=1)
XR = pd.concat([ld("outputs/feat_train.parquet"), ld("outputs/magic_b_train.parquet"),
                ld("outputs/harvest_train.parquet")[json.load(open("outputs/harvest_keep.json"))], N], axis=1)
XR = XR.loc[:, ~XR.columns.duplicated()]
cols = [c for c in XF.columns if c in XR.columns and not (nomb and c.startswith("mb_"))]
sR = ld("outputs/factory/genA.parquet")["g_struct"]
for y in (1, 0):
    r = XR.loc[(lab == y) & sR.isin([0, 1, 2]), cols]
    f = XF.loc[(meta["label"] == y) & sF.isin([0, 1, 2]), cols]
    k = min(len(r), len(f))
    r, f = r.sample(k, random_state=0), f.sample(k, random_state=0)
    X = pd.concat([r, f]).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    z = np.r_[np.zeros(k), np.ones(k)]
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, colsample_bytree=0.5, verbose=-1, n_jobs=16)
    p = cross_val_predict(m, X, z, cv=5, method="predict_proba")[:, 1]
    imp = pd.Series(m.fit(X, z).booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    one = lambda c: (lambda a: max(a, 1 - a))(roc_auc_score(z, X[c].fillna(X[c].median())))
    print(f"\nlabel {y} ({'break' if y else 'no-break, reference'}): {k} real vs {k} fake, real-vs-fake AUC {roc_auc_score(z, p):.3f}")
    for c in imp.index[:12]:
        print(f"   {c:34s} gain share {imp[c] / imp.sum():.3f}  alone AUC {one(c):.3f}  median real {r[c].median():.3g} fake {f[c].median():.3g}")
print(f"{time.time() - t:.0f}s")
