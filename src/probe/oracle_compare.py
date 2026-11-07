"""Practical models on the SAME simplified fakes as the oracle (seeds 0..N-1): LightGBM on the 119 fast features,
train on seeds >= N/2, test on seeds < N/2. Plus our real model (current best OOF) on the real WN / AR1 train series."""
import sys, time, importlib.util
import lightgbm as lgb, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "probe"); import oracle_ceiling as O
t0 = time.time()
FAMS = ["joint_lr", "kind_lr", "scan_lr", "persist_feats", "mosum_feats", "ramp_step", "target_feats", "decoy_feats", "scan_cp"]
def feats(s):
    import importlib.util as iu
    x, n0, q = O.gen(s, bool(s % 2)); out = {}
    for f in FAMS:
        spec = iu.spec_from_file_location(f, f"probe/{f}.py"); m = iu.module_from_spec(spec); spec.loader.exec_module(m)
        try: out.update(m.one(x, n0))
        except Exception: pass
    return out
D = pd.read_parquet("outputs/factory/oracle.parquet"); N = len(D)
X = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(feats)(s) for s in range(N))).astype(float)
te, tr = np.arange(N) < N // 2, np.arange(N) >= N // 2
m = lgb.LGBMClassifier(learning_rate=.05, n_estimators=800, num_leaves=31, min_child_samples=50, subsample=.8, subsample_freq=1,
                       colsample_bytree=.7, verbose=-1, n_jobs=30).fit(X[tr], D.label[tr])
p = m.predict_proba(X[te])[:, 1]; T = D[te].assign(p=p)
print(f"same test fakes: oracle {roc_auc_score(T.label, T.score):.4f} | LGB on 119 fast features {roc_auc_score(T.label, T.p):.4f}  {time.time() - t0:.0f}s")
neg = T[T.label == 0]
for k in ("spread", "level", "dep", "shape"):
    g = pd.concat([neg, T[T.kind == k]]); print(f"  {k:6s} oracle {roc_auc_score(g.label, g.score):.4f}  LGB {roc_auc_score(g.label, g.p):.4f}")
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy()
r = lambda a: rankdata(a) / len(a)
g = np.load("outputs/factory/confirm_combo10gnj.npz")["oof"]; I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy"); best = r(g) + r(I[1]) + r(I[2])
A = pd.read_parquet("outputs/factory/genA.parquet"); A.index = A.index.astype(str); st = A.g_struct.reindex(L.id).to_numpy()
for c, nm in ((0, "WN"), (1, "AR1")):
    k = st == c; print(f"real {nm} train series (n={k.sum()}): our best OOF AUC {roc_auc_score(y[k], best[k]):.4f}")
k = np.isin(st, [0, 1]); print(f"real WN+AR1: our best {roc_auc_score(y[k], best[k]):.4f} | all real {roc_auc_score(y, best):.4f}")
