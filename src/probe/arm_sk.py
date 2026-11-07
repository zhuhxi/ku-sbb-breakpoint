"""sklearn arm on production features: et | knn | lr. 5-fold same split -> outputs/arms/{kind}_oof/test.npy"""
import sys, time, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
sys.path.insert(0, "probe"); import feats
kind = sys.argv[1]; t0 = time.time(); X, Xt, y = feats.load()
A = np.vstack([X.to_numpy(), Xt.to_numpy()]); A = np.where(np.isnan(A), np.nanmedian(A, 0), A)
if kind != "et": A = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=0).fit_transform(A)
Xa, Xb = A[:len(y)], A[len(y):]
mk = dict(et=lambda: ExtraTreesClassifier(500, max_features=.3, min_samples_leaf=5, n_jobs=32, random_state=0),
          knn=lambda: KNeighborsClassifier(200, weights="distance", n_jobs=32),
          lr=lambda: LogisticRegression(C=.01, max_iter=2000))[kind]
oof = np.zeros(len(y)); te = np.zeros(len(Xb))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xa, y)):
    m = mk().fit(Xa[tr], y[tr]); oof[va] = m.predict_proba(Xa[va])[:, 1]; te += m.predict_proba(Xb)[:, 1] / 5
    print(f"fold {k} {roc_auc_score(y[va], oof[va]):.4f} {time.time()-t0:.0f}s", flush=True)
print(f"{kind} OOF {roc_auc_score(y, oof):.4f}"); np.save(f"outputs/arms/{kind}_oof.npy", oof); np.save(f"outputs/arms/{kind}_test.npy", te)
