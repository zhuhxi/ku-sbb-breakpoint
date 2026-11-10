"""Break-specific realism: LightGBM real vs fake on one label class, break-statistic columns (g_ from genA/genB/gnoise/gjlr, optionally saf_).
   python probe/c2st_brk.py [LABEL=1] [PAT=^g_]   -> CV AUC + top features with quantile gaps"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
LAB = int(sys.argv[1]) if len(sys.argv) > 1 else 1; PAT = sys.argv[2] if len(sys.argv) > 2 else "^g_"
E = pd.read_parquet("outputs/aug/extra_gnj_saf.parquet"); E = E.loc[:, E.columns.str.contains(PAT)]
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id").label
M = pd.concat([pd.read_parquet(f"outputs/aug/meta_simr10{b}.parquet") for b in "abef"])
R = E.loc[L.index[L == LAB]]; F = E.loc[M.index[M.label == LAB]].sample(len(R), random_state=0)
X = pd.concat([R, F]).replace([np.inf, -np.inf], np.nan); t = np.r_[np.ones(len(R)), np.zeros(len(F))]; o = np.zeros(len(t)); imp = 0
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, t):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, min_child_samples=50, colsample_bytree=.7, verbose=-1, n_jobs=16).fit(X.iloc[tr], t[tr])
    o[va] = m.predict_proba(X.iloc[va])[:, 1]; imp = imp + m.booster_.feature_importance("gain")
print(f"label {LAB}  n {len(R)}+{len(F)}  {X.shape[1]} cols  real-vs-fake AUC {A(t, o):.4f}")
for c in X.columns[np.argsort(-imp)[:12]]:
    q = [.1, .25, .5, .75, .9]; print(f"  {c:22s} real {np.nanquantile(R[c], q).round(2)}  fake {np.nanquantile(F[c], q).round(2)}")
