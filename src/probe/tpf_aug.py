"""TabPFN with realism-weighted genR10 fakes added to the CONTEXT (fold-clean). Columns = the 841 cols the fakes have + 31 semi-auto
cols (models on bags c,d; out-of-bag for the fakes). Weights: label-free discriminator real (train+test) vs fakes, 2-fold cross-fitted,
w = p/(1-p); fakes enter the context sampled without replacement with prob ~ w.
    python probe/tpf_aug.py K NF [folds...]     env FAKES (default a,b,e,f), NE, TAG (saves outputs/arms/tpfaug{TAG}_f{k}.npz)"""
import json, os, sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
t0 = time.time(); K, NF = int(sys.argv[1]), int(sys.argv[2]); FOLDS = [int(v) for v in sys.argv[3:]] or list(range(5))
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]; sub_ids = pd.read_csv("data/sample_submission.csv", dtype={"id": str})["id"]
FAM = ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")
def frame(idx, part):
    ld = lambda p: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    N = pd.concat([ld(f"outputs/factory/{f}.parquet") for f in FAM], axis=1)
    N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$", regex=True)]
    X = pd.concat([ld(f"outputs/feat_{part}.parquet"), ld(f"outputs/harvest_{part}.parquet")[json.load(open("outputs/harvest_keep.json"))], N.loc[:, ~N.columns.duplicated()]], axis=1)
    return X.loc[:, ~X.columns.duplicated()]
XR, XT = frame(lab.index, "train"), frame(sub_ids, "test")
FK = ["simr10" + b for b in os.environ.get("FAKES", "abef")]
XS = pd.concat([pd.read_parquet(f"outputs/aug/X_{f}.parquet") for f in FK]); ys = pd.concat([pd.read_parquet(f"outputs/aug/meta_{f}.parquet") for f in FK]).loc[XS.index, "label"].to_numpy()
E = pd.read_parquet("outputs/aug/saf_cd_xabef.parquet"); XR, XT, XS = [pd.concat([A, E.reindex(A.index)], axis=1) for A in (XR, XT, XS)]
cols = [c for c in XR.columns if c in XS.columns and not c.startswith("mb_")]
XR, XT, XS = [A[cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32) for A in (XR, XT, XS)]
y = lab.to_numpy(); print(f"{len(cols)} cols, fakes {len(ys)}  {time.time()-t0:.0f}s", flush=True)
# label-free realism weights (real train+test vs fakes)
w = np.zeros(len(ys)); RA = np.r_[XR, XT]
for a, b in KFold(2, shuffle=True, random_state=0).split(XS):
    p = lgb.LGBMClassifier(n_estimators=150, learning_rate=.08, num_leaves=31, colsample_bytree=.3, subsample=.8, subsample_freq=1, min_child_samples=50, verbose=-1, n_jobs=16).fit(np.r_[RA, XS[a]], np.r_[np.ones(len(RA)), np.zeros(len(a))]).predict_proba(XS[b])[:, 1]
    w[b] = p / (1 - p + 1e-6)
w = np.clip(w, 0, 20); print(f"weights q10/50/90 {np.quantile(w, [.1, .5, .9]).round(3).tolist()}  {time.time()-t0:.0f}s", flush=True)
from tabpfn import TabPFNClassifier
rng = np.random.default_rng(0); res = []
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(XR, y)):
    if k not in FOLDS: continue
    a_, b_ = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    g = lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=31, colsample_bytree=.5, subsample=.8, subsample_freq=1, min_child_samples=40, verbose=-1, n_jobs=16).fit(XR[a_], y[a_])
    top = np.argsort(-g.booster_.feature_importance("gain"))[:K]
    fi = rng.choice(len(ys), NF, replace=False, p=w / w.sum()) if NF else np.array([], int)
    Xc, yc = np.r_[XR[tr][:, top], XS[fi][:, top]], np.r_[y[tr], ys[fi]]
    FLAG = bool(os.environ.get("FLAG")); fl = lambda A, v: np.c_[A, np.full(len(A), v, np.float32)] if FLAG else A
    if FLAG: Xc = np.r_[fl(XR[tr][:, top], 1), fl(XS[fi][:, top], 0)]
    m = TabPFNClassifier(device="cuda", n_estimators=int(os.environ.get("NE", 4)), ignore_pretraining_limits=True, random_state=0).fit(Xc, yc)
    pv = m.predict_proba(fl(XR[va][:, top], 1))[:, 1]; res.append(roc_auc_score(y[va], pv))
    if os.environ.get("TAG"): np.savez(f"outputs/arms/tpfaug{os.environ['TAG']}_f{k}.npz", va=pv, idx=va, test=m.predict_proba(fl(XT[:, top], 1))[:, 1])
    print(f"fold {k}: K {K} NF {NF} ctx {len(yc)} AUC {res[-1]:.4f}  {time.time()-t0:.0f}s", flush=True)
print(f"mean {np.mean(res):.4f}")
