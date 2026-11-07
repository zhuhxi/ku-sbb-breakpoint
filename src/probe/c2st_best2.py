"""Where is the best2 generator still wrong? Classifier two-sample test (C2ST) real train vs best2 fakes on the FULL feature set
(841 columns, outputs/aug/X_simk2.parquet), separately per label (real label y vs fake label y). Reports C2ST AUC (.5 = perfect
generator), the most separating features (gain + per-feature KS), feature-family shares of the gain, and which fake families /
break kinds look most fake (mean out-of-fold fake score).        python probe/c2st_best2.py"""
import json, re, time
import lightgbm as lgb, numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
t0 = time.time()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
FAM = ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")
ld = lambda p: pd.read_parquet(p).reindex(lab.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
N = pd.concat([ld(f"outputs/factory/{f}.parquet") for f in FAM], axis=1)
N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$", regex=True)]
XR = pd.concat([ld("outputs/feat_train.parquet"), ld("outputs/harvest_train.parquet")[json.load(open("outputs/harvest_keep.json"))], N.loc[:, ~N.columns.duplicated()]], axis=1)
XR = XR.loc[:, ~XR.columns.duplicated()]
XS = pd.read_parquet("outputs/aug/X_simk2.parquet"); H = pd.read_parquet("outputs/aug/meta_simk2.parquet").loc[XS.index]
cols = [c for c in XR.columns if c in XS.columns and not c.startswith("mb_")]
XR, XS = XR[cols], XS[cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
print(f"{len(cols)} columns  load {time.time() - t0:.0f}s", flush=True)
fam_of = lambda c: (re.match(r"^([a-z0-9]+)_", c).group(1) if re.match(r"^([a-z0-9]+)_", c) else "other")
for y in (0, 1):
    r, f = XR[lab.to_numpy() == y], XS[H.label.to_numpy() == y]; hf = H[H.label == y]
    X = pd.concat([r, f]); t = np.r_[np.zeros(len(r)), np.ones(len(f))]
    oof, gain = np.zeros(len(t)), np.zeros(len(cols))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, t):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=31, colsample_bytree=.5, subsample=.8, subsample_freq=1, verbose=-1, n_jobs=30).fit(X.iloc[tr], t[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; gain += m.booster_.feature_importance("gain")
    print(f"\n=== label {y}: real {len(r)} vs fake {len(f)}   C2ST AUC {roc_auc_score(t, oof):.4f}   {time.time() - t0:.0f}s")
    G = pd.Series(gain / gain.sum(), index=cols).sort_values(ascending=False)
    top = G.index[:12]
    for c in top:
        a, b = r[c].dropna(), f[c].dropna()
        print(f"  {c:40s} gain {G[c]:.3f}  KS {ks_2samp(a, b).statistic:.3f}  median real {a.median():+.3g} fake {b.median():+.3g}")
    print("  gain share by feature family:", G.groupby([fam_of(c) for c in G.index]).sum().sort_values(ascending=False).head(8).round(3).to_dict())
    fo = pd.Series(oof[len(r):], index=f.index)
    print("  mean fake-score of fakes by generator family:", fo.groupby(hf.family).mean().round(3).sort_values(ascending=False).to_dict())
    if y == 1:
        print("  ... by break kind:", fo.groupby(hf.kind).mean().round(3).sort_values(ascending=False).to_dict())
