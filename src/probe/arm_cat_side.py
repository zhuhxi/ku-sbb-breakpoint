"""CatBoost on the exact inputs of the best TabPFN arm: per fold top-K gain features (quick LGB on the inner-train part; bag6w set
+ seqembmtw), raw median-filled values of the original row AND of its reversed series side by side. CPU.
    python probe/arm_cat_side.py K TAG [depth]"""
import sys, os, time, numpy as np, lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
sys.path.insert(0, "probe"); import feats
K, TAG = int(sys.argv[1]), sys.argv[2]; D = int(sys.argv[3]) if len(sys.argv) > 3 else 6; REV = "/mnt/disk2/zhuhx/kagglecompetition/ku_sbb_rev"; t0 = time.time()
feats.FAMS[-1] = "semiauto_bag6w"; feats.FAMS += ["seqembmtw"]
X, Xt, y = feats.load(); cwd = os.getcwd(); os.chdir(REV); XR, XTR, _ = feats.load(); os.chdir(cwd)
XR, XTR = XR[X.columns].to_numpy(np.float32), XTR[X.columns].to_numpy(np.float32); Xn, XT = X.to_numpy(np.float32), Xt.to_numpy(np.float32)
ALL = np.vstack([Xn, XT, XR, XTR]); ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL); n, m = len(Xn), len(XT)
Q = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=2000, verbose=-1, n_jobs=30)
oof = np.zeros(n); te = np.zeros(m)
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xn, y)):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**Q, random_state=0).fit(Xn[a], y[a], eval_set=[(Xn[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    top = np.argsort(-m0.booster_.feature_importance("gain"))[:K]; Z = np.hstack([ALL[:n + m, top], ALL[n + m:, top]]); Ztr, Zte = Z[:n], Z[n:]
    c = CatBoostClassifier(iterations=3000, learning_rate=.03, depth=D, l2_leaf_reg=3, eval_metric="AUC", od_type="Iter", od_wait=200, thread_count=30, verbose=0, random_seed=0)
    c.fit(Ztr[a], y[a], eval_set=(Ztr[b], y[b])); oof[va] = c.predict_proba(Ztr[va])[:, 1]; te += c.predict_proba(Zte)[:, 1] / 5
    print(f"fold {k}: {roc_auc_score(y[va], oof[va]):.4f} (iters {c.get_best_iteration()}) {time.time()-t0:.0f}s", flush=True)
print(f"{TAG} OOF {roc_auc_score(y, oof):.4f}"); np.save(f"outputs/arms/{TAG}_oof.npy", oof); np.save(f"outputs/arms/{TAG}_test.npy", te)
