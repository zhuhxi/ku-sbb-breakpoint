"""Family specialist: TabPFN trained only on one semi-auto family (fold-clean), then re-order that family's stacker scores
(quantile mapping keeps the family's score distribution). python probe/spec_rw.py STACK_OOF FAM K W..."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "probe"); import feats
from scipy.stats import rankdata as r
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
o = np.load(sys.argv[1]); FAM = sys.argv[2]; K = int(sys.argv[3]); Ws = [float(w) for w in sys.argv[4:]]
X, _, y = feats.load(); SA = pd.read_parquet("outputs/factory/semiauto_bag8r.parquet").reindex(X.index)
fam = [c for c in SA if c.startswith("saf_family_")]; m = np.array([fam[i][11:] for i in SA[fam].values.argmax(1)]) == FAM
Xm, ym = X.values[m].astype(np.float32), y[m]; sp = np.zeros(m.sum())
for tr, va in StratifiedKFold(5, shuffle=True, random_state=11).split(Xm, ym):
    g = lgb.LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=15, min_child_samples=20, colsample_bytree=.5, verbose=-1, n_jobs=16).fit(Xm[tr], ym[tr])
    top = np.argsort(-g.booster_.feature_importance("gain"))[:K]
    sp[va] = TabPFNClassifier(device="cuda", n_estimators=8, ignore_pretraining_limits=True).fit(Xm[tr][:, top], ym[tr]).predict_proba(Xm[va][:, top])[:, 1]
print(f"{FAM} n {m.sum()} stack {A(ym, o[m]):.4f} specialist {A(ym, sp):.4f} overall {A(y, o):.4f}")
for w in Ws:
    s = r(o[m]) + w * r(sp); o2 = o.copy(); o2[np.where(m)[0][np.argsort(s)]] = np.sort(o[m])
    print(f"  w {w}: {FAM} {A(ym, s):.4f} overall {A(y, o2):.4f}")
