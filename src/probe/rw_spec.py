"""RW specialist: TabPFN trained on RW-family series only (family = argmax of semiauto family probs), features: top-K by a quick LGB
on the fold's RW training rows (+ the reversed-series copy of those columns side by side). Folds = the global 5 folds (seed 42)."""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "probe"); import feats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
feats.FAMS += os.environ.get("EXTRA", "seqembmtw").split(","); X, XT, _ = feats.load(); y = feats.y; K = int(sys.argv[1]); FAMQ = sys.argv[2] if len(sys.argv) > 2 else "RW"
A = pd.read_parquet("outputs/factory/semiauto_bag6w.parquet"); fc = [c for c in A if c.startswith("saf_family_")]
KC = [c for c in A if c.startswith("saf_kind_")]
fam = (lambda ids: A.reindex(ids)[KC].to_numpy().argmax(1) == KC.index(f"saf_kind_{FAMQ}")) if os.environ.get("BYKIND") else (lambda ids: A.reindex(ids)[fc].to_numpy().argmax(1) == fc.index(f"saf_family_{FAMQ}"))
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); m = fam(L.id); Xn = X.to_numpy(np.float32); med = np.nanmedian(Xn, 0); Xn = np.where(np.isnan(Xn), med, Xn)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)); o = np.full(len(y), np.nan)
for k, (tr, va) in enumerate(folds):
    a = tr[m[tr]]; b = va[m[va]]
    q = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, min_child_samples=20, colsample_bytree=.5, verbose=-1, n_jobs=16).fit(Xn[a], y[a])
    top = np.argsort(-q.booster_.feature_importance("gain"))[:K]
    o[b] = TabPFNClassifier(device="cuda", n_estimators=8, ignore_pretraining_limits=True, random_state=0).fit(Xn[a][:, top], y[a]).predict_proba(Xn[b][:, top])[:, 1]
    print(f"fold {k}: n_tr {len(a)} AUC {roc_auc_score(y[b], o[b]):.4f}", flush=True)
ref = np.load(os.environ.get("REF", "outputs/arms/tpf60siderw0_oof.npy"))
print(f"{FAMQ} specialist K{K}: {roc_auc_score(y[m], o[m]):.4f} | global best arm on {FAMQ}: {roc_auc_score(y[m], ref[m]):.4f} | avg {roc_auc_score(y[m], pd.Series(o[m]).rank().values + pd.Series(ref[m]).rank().values):.4f}")
np.save(f"outputs/arms/spec{FAMQ}{K}_oof.npy", o); oo = np.where(np.isnan(o), ref, o); print("filled OOF (spec on subset, global elsewhere)", round(roc_auc_score(y, pd.Series(oo).rank().values), 4))
