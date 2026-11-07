"""Generator -> break posterior. Fakes from the clean generator (probe/ku_gen.py, search-best parameters) get the
same compact feature set as the real series (nine fast families, one(x, n0) each); a LightGBM trained on FAKES only
predicts the real series.
  TSTR = AUC of the fake-trained model on the real training labels (vs a real-trained model on the same features, 5-fold)
  the fake-trained probability is saved as a feature for the real model: outputs/factory/simpost.parquet

    python probe/sim_post.py NFAKE
"""
import importlib.util
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, "probe")
NF = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
sys.argv = sys.argv[:1]
from ku_gen import DEFAULT, gen

FAMS = ["joint_lr", "kind_lr", "scan_lr", "persist_feats", "mosum_feats", "ramp_step", "target_feats", "decoy_feats", "scan_cp"]
mods = {}
for f in FAMS:
    spec = importlib.util.spec_from_file_location(f, f"probe/{f}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods[f] = m


def feats(x, n0):
    out = {}
    for f in FAMS:
        try:
            out.update(mods[f].one(x, n0))
        except Exception:
            pass
    return out


t0 = time.time()
S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
items = [(str(k), g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
XR = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(feats)(v, n0) for _, v, n0 in items), index=[k for k, _, _ in items]).astype(float)
print(f"real features {XR.shape} {time.time() - t0:.0f}s", flush=True)
b = {k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v) for k, v in json.load(open("outputs/factory/ku_gen_best.json")).items()}
P = dict(DEFAULT, **b)
ys = np.arange(NF) % 2


def fake(s):
    x, n0, _ = gen(P, 3_000_000 + s, bool(s % 2))
    return feats(x, n0)


XF = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(fake)(s) for s in range(NF))).astype(float)
XF = XF.reindex(columns=XR.columns)
print(f"fake features {XF.shape} {time.time() - t0:.0f}s", flush=True)
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
Xtr, y = XR.loc[L.index], L.to_numpy()
prm = dict(objective="binary", learning_rate=.05, num_leaves=31, min_child_samples=50, subsample=.8, subsample_freq=1,
           colsample_bytree=.7, n_estimators=600, verbose=-1, n_jobs=30)
m = lgb.LGBMClassifier(**prm).fit(XF, ys)
p_real = m.predict_proba(XR)[:, 1]
tstr = roc_auc_score(y, pd.Series(p_real, index=XR.index).loc[L.index])
o = cross_val_predict(lgb.LGBMClassifier(**prm), Xtr, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
print(f"TSTR (train on {NF} fakes, test on real train labels) AUC {tstr:.4f} | real-trained 5-fold on the same {XR.shape[1]} features {roc_auc_score(y, o):.4f}")
fo = cross_val_predict(lgb.LGBMClassifier(**prm), XF, ys, cv=5, method="predict_proba")[:, 1]
print(f"fake-on-fake 5-fold AUC {roc_auc_score(ys, fo):.4f}  (gap to TSTR = what the generator still gets wrong)")
pd.DataFrame({"sim_post": p_real, "sim_logit": np.log(np.clip(p_real, 1e-6, 1 - 1e-6) / np.clip(1 - p_real, 1e-6, 1))}, index=XR.index).to_parquet("outputs/factory/simpost.parquet")
XR.to_parquet("outputs/factory/fastfam_real.parquet")
print(f"done {time.time() - t0:.0f}s")
