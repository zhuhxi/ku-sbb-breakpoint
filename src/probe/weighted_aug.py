"""Importance-weighted fake augmentation (density-ratio weights from a real-vs-fake discriminator).
Real train (24k) and 30k best2 fakes (outputs/aug/X_simk2.parquet) on the 841 shared columns. Inside each outer fold (same
StratifiedKFold(5, shuffle, 42)), using ONLY the outer-train real rows: per class, a LightGBM discriminator real vs fake,
cross-fitted over the fakes (2 folds) -> w = p(real|x) / p(fake|x), clipped to [0, 5], scaled to mean W0 (real rows weight 1).
Arms (same LightGBM, lr .1, early stopping on an inner 10% real split): real only | + fakes, flat weight W0 | + fakes, ratio
weights mean W0. Paired bootstrap of the OOF AUC vs real only.        python probe/weighted_aug.py [W0 [EXTRA.parquet]]
EXTRA: more columns for real AND fake rows (indexed by real id and fake id), e.g. out-of-bag semi-auto estimates."""
import json, sys, time
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
t0 = time.time(); W0 = float(sys.argv[1]) if len(sys.argv) > 1 else .3; EXTRA = sys.argv[2] if len(sys.argv) > 2 else None
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
FAM = ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")
sub_ids = pd.read_csv("data/sample_submission.csv", dtype={"id": str})["id"]
def frame(idx, part):
    ld = lambda p: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    N = pd.concat([ld(f"outputs/factory/{f}.parquet") for f in FAM], axis=1)
    N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$", regex=True)]
    X = pd.concat([ld(f"outputs/feat_{part}.parquet"), ld(f"outputs/harvest_{part}.parquet")[json.load(open("outputs/harvest_keep.json"))], N.loc[:, ~N.columns.duplicated()]], axis=1)
    return X.loc[:, ~X.columns.duplicated()]
XR = frame(lab.index, "train"); XT = frame(sub_ids, "test")
import os; FK = os.environ.get("FAKES", "simk2").split(",")   # one or more fake bags in outputs/aug
XS = pd.concat([pd.read_parquet(f"outputs/aug/X_{f}.parquet") for f in FK]); ys = pd.concat([pd.read_parquet(f"outputs/aug/meta_{f}.parquet") for f in FK]).loc[XS.index, "label"].to_numpy()
if os.environ.get("NFK"): XS, ys = XS.iloc[:int(os.environ["NFK"])], ys[:int(os.environ["NFK"])]
if EXTRA:
    E = pd.read_parquet(EXTRA); XR = pd.concat([XR, E.reindex(XR.index)], axis=1); XT = pd.concat([XT, E.reindex(XT.index)], axis=1); XS = pd.concat([XS, E.reindex(XS.index)], axis=1)
    print(f"extra {E.shape[1]} cols, fake rows covered {E.index.isin(XS.index).sum()}")
cols = [c for c in XR.columns if c in XS.columns and not c.startswith("mb_")]
XR, XT, XS = XR[cols].to_numpy(np.float32), XT[cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32), XS[cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
y = lab.to_numpy(); print(f"{len(cols)} cols, real {len(y)}, fakes {len(ys)}  {time.time() - t0:.0f}s", flush=True)
D = dict(n_estimators=150, learning_rate=.08, num_leaves=31, colsample_bytree=.3, subsample=.8, subsample_freq=1, min_child_samples=50, verbose=-1, n_jobs=30)
M = dict(objective="binary", learning_rate=.1, num_leaves=31, min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.5, reg_lambda=1., n_estimators=2000, verbose=-1, n_jobs=30)
arms = ("real only", "+ fakes flat", "+ fakes weighted"); SAVE = os.environ.get("SAVE"); ptest = np.zeros(len(XT))
oof = {a: np.zeros(len(y)) for a in arms}
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(XR, y)):
    w = np.zeros(len(ys))
    for c in (0, 1):
        R, fi = XR[tr][y[tr] == c], np.flatnonzero(ys == c)
        for a, b in KFold(2, shuffle=True, random_state=k).split(fi):
            X = np.r_[R, XS[fi[a]]]; t = np.r_[np.ones(len(R)), np.zeros(len(a))]
            p = lgb.LGBMClassifier(**D).fit(X, t).predict_proba(XS[fi[b]])[:, 1]
            w[fi[b]] = (p / (1 - p + 1e-6)) * (len(a) / len(R))
    w = np.clip(w, 0, 5); w = w / w.mean() * W0
    a_, b_ = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    for arm in arms:
        if arm == "real only":
            X, t, sw = XR[a_], y[a_], np.ones(len(a_))
        else:
            X, t = np.r_[XR[a_], XS], np.r_[y[a_], ys]
            sw = np.r_[np.ones(len(a_)), np.full(len(ys), W0) if arm == "+ fakes flat" else w]
        m = lgb.LGBMClassifier(**M, random_state=0).fit(X, t, sample_weight=sw, eval_set=[(XR[b_], y[b_])], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[arm][va] = m.predict_proba(XR[va])[:, 1]
        if SAVE and arm == "+ fakes weighted": ptest += m.predict_proba(XT)[:, 1] / 5
    print(f"fold {k}: " + "  ".join(f"{a} {roc_auc_score(y[va], oof[a][va]):.4f}" for a in arms) + f"  | weight q10/50/90 {np.quantile(w, [.1, .5, .9]).round(2).tolist()}  {time.time() - t0:.0f}s", flush=True)
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(300)]
for a in arms[1:]:
    d = [roc_auc_score(y[i], oof[a][i]) - roc_auc_score(y[i], oof["real only"][i]) for i in B]
    print(f"{a:18s} OOF {roc_auc_score(y, oof[a]):.4f} vs real only {roc_auc_score(y, oof['real only']):.4f}: {roc_auc_score(y, oof[a]) - roc_auc_score(y, oof['real only']):+.4f} CI [{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}]")
if SAVE: np.save(f"outputs/arms/{SAVE}_oof.npy", oof["+ fakes weighted"]); np.save(f"outputs/arms/{SAVE}_test.npy", ptest); np.save(f"outputs/arms/{SAVE}real_oof.npy", oof["real only"])
print(f"{time.time() - t0:.0f}s")
