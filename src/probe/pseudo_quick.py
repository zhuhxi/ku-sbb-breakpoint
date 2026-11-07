"""Pseudo-label check (fold-clean): per fold, teacher = quick LGB on fold-train; test rows added with SOFT labels
(each test row twice: label 1 weight p, label 0 weight 1-p), times PW. Compare student vs teacher on the val fold."""
import sys, time, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
PW = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0; MODE = sys.argv[2] if len(sys.argv) > 2 else "soft"
t0 = time.time(); X, Xt, y = feats.load(); X, Xt = X.to_numpy(), Xt.to_numpy()
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=32)
oof = {"teacher": np.zeros(len(y)), "student": np.zeros(len(y))}
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    a, b = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=k)
    m = lgb.LGBMClassifier(**P, random_state=int(__import__("os").environ.get("SEED", 0))).fit(X[a], y[a], eval_set=[(X[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    oof["teacher"][va] = m.predict_proba(X[va])[:, 1]; p = m.predict_proba(Xt)[:, 1]
    if MODE == "soft":
        XX = np.vstack([X[a], Xt, Xt]); yy = np.r_[y[a], np.ones(len(Xt)), np.zeros(len(Xt))]; ww = np.r_[np.ones(len(a)), PW * p, PW * (1 - p)]
    else:  # hard: confident tails only
        q = float(MODE); sel = (p > np.quantile(p, 1 - q)) | (p < np.quantile(p, q))
        XX = np.vstack([X[a], Xt[sel]]); yy = np.r_[y[a], (p[sel] > .5).astype(float)]; ww = np.r_[np.ones(len(a)), PW * np.ones(sel.sum())]
    s = lgb.LGBMClassifier(**P, random_state=int(__import__("os").environ.get("SEED", 0))).fit(XX, yy, sample_weight=ww, eval_set=[(X[b], y[b])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    oof["student"][va] = s.predict_proba(X[va])[:, 1]
    print(f"fold {k}: teacher {roc_auc_score(y[va], oof['teacher'][va]):.4f} student {roc_auc_score(y[va], oof['student'][va]):.4f} {time.time()-t0:.0f}s", flush=True)
np.save(f"outputs/arms/pseudo_{MODE}_{PW}_oof.npy", oof["student"]); np.save(f"outputs/arms/pseudo_teacher_oof.npy", oof["teacher"])
rng = np.random.default_rng(0)
d = [roc_auc_score(y[i], oof["student"][i]) - roc_auc_score(y[i], oof["teacher"][i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(200))]
print(f"PW {PW} {MODE}: teacher {roc_auc_score(y, oof['teacher']):.4f} student {roc_auc_score(y, oof['student']):.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]")
