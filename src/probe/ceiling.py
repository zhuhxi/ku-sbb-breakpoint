"""Ceiling / learning curve on genR10 fakes: LightGBM trained on N fakes (903 cols incl. gnj + semi-auto c,d), tested on held-out bag f.
Also real->real (5-fold, same LGB) and fake->real (TSTR) for reference.   python probe/ceiling.py"""
import json, numpy as np, pandas as pd, lightgbm as lgb, time
from sklearn.metrics import roc_auc_score as A
from sklearn.model_selection import StratifiedKFold, train_test_split
t0 = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
FAM = ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")
def frame(idx, part):
    ld = lambda p: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    N = pd.concat([ld(f"outputs/factory/{f}.parquet") for f in FAM], axis=1)
    N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$", regex=True)]
    X = pd.concat([ld(f"outputs/feat_{part}.parquet"), ld(f"outputs/harvest_{part}.parquet")[json.load(open("outputs/harvest_keep.json"))], N.loc[:, ~N.columns.duplicated()]], axis=1)
    return X.loc[:, ~X.columns.duplicated()]
XR = frame(lab.index, "train"); E = pd.read_parquet("outputs/aug/extra_gnj_saf.parquet")
B = {b: pd.read_parquet(f"outputs/aug/X_simr10{b}.parquet") for b in "abef"}; Y = {b: pd.read_parquet(f"outputs/aug/meta_simr10{b}.parquet").loc[B[b].index, "label"].to_numpy() for b in "abef"}
XR = pd.concat([XR, E.reindex(XR.index)], axis=1); B = {b: pd.concat([v, E.reindex(v.index)], axis=1) for b, v in B.items()}
cols = [c for c in XR.columns if c in B["a"].columns and not c.startswith("mb_")]
XR = XR[cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32); B = {b: v[cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32) for b, v in B.items()}; y = lab.to_numpy()
P = dict(objective="binary", learning_rate=.05, num_leaves=31, min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.5, reg_lambda=1., n_estimators=3000, verbose=-1, n_jobs=30)
def fit(X, t):
    a, b = train_test_split(np.arange(len(t)), test_size=.1, stratify=t, random_state=0)
    return lgb.LGBMClassifier(**P).fit(X[a], t[a], eval_set=[(X[b], t[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
print(len(cols), "cols", f"{time.time()-t0:.0f}s", flush=True)
XF, YF = np.r_[B["a"], B["b"], B["e"]], np.r_[Y["a"], Y["b"], Y["e"]]; XH, YH = B["f"], Y["f"]
for n in (6000, 19200, 45000, 90000):
    m = fit(XF[:n], YF[:n]); print(f"fake->fake train {n:6d}: test AUC {A(YH, m.predict_proba(XH)[:, 1]):.4f}   fake->real (TSTR) {A(y, m.predict_proba(XR)[:, 1]):.4f}  {time.time()-t0:.0f}s", flush=True)
o = np.zeros(len(y))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(XR, y): o[va] = fit(XR[tr], y[tr]).predict_proba(XR[va])[:, 1]
print(f"real->real 5-fold (train 19200): {A(y, o):.4f}  {time.time()-t0:.0f}s")
