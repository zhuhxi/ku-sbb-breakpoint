"""Adversarial validation: can LGB tell train from test on the production features?"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "probe"); import feats
X, Xt, y = feats.load(); A = pd.concat([X, Xt]); t = np.r_[np.zeros(len(X)), np.ones(len(Xt))]
m = lgb.LGBMClassifier(n_estimators=150, learning_rate=.05, num_leaves=31, colsample_bytree=.5, verbose=-1, n_jobs=32)
p = cross_val_predict(m, A.to_numpy(np.float32), t, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"train-vs-test AUC {roc_auc_score(t, p):.4f}")
m.fit(A.to_numpy(np.float32), t); imp = pd.Series(m.booster_.feature_importance("gain"), A.columns).sort_values()[::-1]
print((imp / imp.sum()).head(10).round(3).to_string())
for c in imp.index[:5]: print(c, "train median", round(float(X[c].median()), 3), "test", round(float(Xt[c].median()), 3))
