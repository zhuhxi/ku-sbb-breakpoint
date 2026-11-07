"""Top features by gain of a fast LightGBM on the combo10g columns (fold 0 train part, lr 0.1).

    python probe/top_feats.py [K]
"""
import sys
import time

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.argv = [sys.argv[0], "x"] + sys.argv[1:]
t = time.time()
import importlib.util
src = open("probe/param_quick.py").read().split("tr, va = next")[0].replace("name, vals = sys.argv[1], [float(v) for v in sys.argv[2:]]", "")
exec(src)
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
tr, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
m = lgb.LGBMClassifier(objective="binary", learning_rate=0.1, n_estimators=150, num_leaves=31, min_child_samples=40,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=30, random_state=0).fit(X.iloc[tr], y[tr])
imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print((imp / imp.sum()).head(K).round(3).to_string())
print(f"{time.time() - t:.1f}s")
