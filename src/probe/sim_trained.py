"""How useful is the current generator for the real task? Train LightGBM only on generated series (menu
"L abc40 ns35", half with a break) and score real labelled series; compare with the same features trained on real
labels (5-fold CV). Features: before-vs-after statistics (probe/hidden_diag.py) + break sizes (probe/model_feats.py).

    python probe/sim_trained.py
"""
import sys
import time

sys.argv = [sys.argv[0], "--menu", "L abc40 ns35", "--only-break", "1"]
import importlib.util
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
bc = load("bc", "probe/break_c2st.py")
sys.argv = [sys.argv[0]]
mf = load("mf", "probe/model_feats.py")
NSIM, NREAL = 8000, 4000


def feats(v, n0):
    f = bc.feats(v, n0)
    f.update(mf.gfeats(v, n0))
    return f


def sim(seed, brk):
    x, n0, _ = bc.cb.gen(bc.MENU, seed, brk, "truth")
    return feats(x, n0)


t0 = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = np.random.default_rng(3).choice(lab.index, NREAL, False)
S = S[S.id.isin(set(ids))].sort_values(["id", "time"])
real = [(i, g["value"].to_numpy(), int((g["period"] == 0).sum())) for i, g in S.groupby("id")]
R = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(feats)(v, n0) for _, v, n0 in real))
yr = lab.loc[[i for i, _, _ in real]].to_numpy()
G = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(sim)(9 * 10 ** 6 + i, i % 2 == 1) for i in range(NSIM)))
yg = np.arange(NSIM) % 2
print(f"features done ({time.time() - t0:.0f}s)")
clean = lambda D: D.drop(columns="cls").rename(columns=lambda c: "".join(ch if ch.isalnum() else "_" for ch in c)).astype(float)
XR, XG = clean(R), clean(G)
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=30)
p_sim = m.fit(XG, yg).predict_proba(XR)[:, 1]
p_real = cross_val_predict(m, XR, yr, cv=5, method="predict_proba")[:, 1]
ws = R.cls.isin(["WN", "AR1", "AR2"]).to_numpy()
print(f"trained on generated only -> real AUC {roc_auc_score(yr, p_sim):.4f}   (WN/AR pre-structure only: {roc_auc_score(yr[ws], p_sim[ws]):.4f})")
print(f"trained on real labels    -> real AUC {roc_auc_score(yr, p_real):.4f}   (WN/AR pre-structure only: {roc_auc_score(yr[ws], p_real[ws]):.4f})")
p_both = cross_val_predict(m, XR.assign(sim_score=p_sim), yr, cv=5, method="predict_proba")[:, 1]
print(f"real labels + sim score   -> real AUC {roc_auc_score(yr, p_both):.4f}")
print(f"total {time.time() - t0:.0f}s")
