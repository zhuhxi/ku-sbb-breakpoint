"""Step 1: at which jump sizes does the real-vs-generated classifier catch the generated break series?
Out-of-fold fake probability and counts, bucketed by the measured level change (noise-sd units).

    python probe/c2st_by_level.py MENU
"""
import importlib.util
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")
name = sys.argv[1]
sys.argv = [sys.argv[0], "--n", "1"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
c2 = load("c2", "probe/break_c2st.py")
bm = load("bm", "probe/break_measure.py")

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = set(np.random.default_rng(21).choice(lab.index[lab == 1], 3000, False))
real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
fake = Parallel(n_jobs=30)(delayed(c2.cb.gen)(c2.cb.BREAKS[name], 11 * 10 ** 6 + s, True, True) for s in range(3000))
F = pd.DataFrame(Parallel(n_jobs=30)(delayed(c2.feats)(v, n0) for v, n0 in real + fake))
F["lvl"] = [bm.measure(v, n0)[0] for v, n0 in real + fake]
F["y"] = np.r_[np.zeros(len(real)), np.ones(len(fake))]
F = F[F.cls.isin(["WN", "AR1", "AR2"])]
X = F.drop(columns=["cls", "y", "lvl"])
X.columns = [c.replace("(", "").replace(")", "").replace(">", "gt").replace(":", "") for c in X.columns]
F["p"] = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=30),
                           X, F.y, cv=5, method="predict_proba")[:, 1]
F["bucket"] = pd.cut(F.lvl.abs(), [0, .1, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2.25, 2.75, 3.5, 99], right=False)
T = F.groupby(["bucket", "y"], observed=True).agg(n=("p", "size"), p_fake=("p", "mean")).unstack()
T.columns = ["n real", "n gen", "p_fake real", "p_fake gen"]
T["gen/real"] = T["n gen"] / T["n real"] * (F.y == 0).sum() / (F.y == 1).sum()
print(T.round(2).to_string())
