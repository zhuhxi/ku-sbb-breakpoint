"""Semi-automatic ABC on the FULL feature set: hidden-parameter models trained on best2 fakes with the real model's columns
(outputs/aug/X_TAG.parquet from aug_features.py, meta with hidden parameters from probe/sim_series_ku2.py), same targets as
probe/semiauto.py v2; fitted values on real train + test -> outputs/factory/semiauto_full.parquet.
    python probe/semiauto_full.py TAG [OUT [XTRA_TAG]]
"""
import json, os, sys, time
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

t0 = time.time(); TAG = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/factory/semiauto_full.parquet"
XTRA = sys.argv[3] if len(sys.argv) > 3 else None   # another fake tag: also score those fakes (never trained on)
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
FAM = ("st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost")


def frame(idx, part):
    ld = lambda p: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    N = pd.concat([ld(f"outputs/factory/{f}.parquet") for f in FAM], axis=1)
    N = N.loc[:, N.columns.str.contains(r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$", regex=True)]
    X = pd.concat([ld(f"outputs/feat_{part}.parquet"), ld(f"outputs/harvest_{part}.parquet")[json.load(open("outputs/harvest_keep.json"))],
                   N.loc[:, ~N.columns.duplicated()]], axis=1)
    return X.loc[:, ~X.columns.duplicated()]


XS = pd.read_parquet(f"outputs/aug/X_{TAG}.parquet"); H = pd.read_parquet(f"outputs/aug/meta_{TAG}.parquet").loc[XS.index]
XR = pd.concat([frame(lab.index, "train"), frame(sub["id"], "test")])
if XTRA:
    XR = pd.concat([XR, pd.read_parquet(f"outputs/aug/X_{XTRA}.parquet")[[c for c in XR.columns if c in XS.columns]]])
cols = [c for c in XR.columns if c in XS.columns and not c.startswith("mb_")]
XS, XR = XS[cols].replace([np.inf, -np.inf], np.nan).astype(np.float32), XR[cols]
print(f"{len(cols)} columns, {len(XS)} fakes, load {time.time() - t0:.0f}s", flush=True)
KINDS, FAMI = [None, "spread", "level", "dep", "shape"], ["WN", "AR1", "AR2", "SIN", "RW", "LL"]
T = {"kind": H.kind.map({k: i for i, k in enumerate(KINDS)}).fillna(0).to_numpy(int), "family": H.family.map({k: i for i, k in enumerate(FAMI)}).to_numpy(int)}
for c in ("log_k", "lvl_d", "dep_d", "p1", "p2", "inv_df", "sin_amp"): T[c] = H[c].to_numpy(float)
T["abs_log_k"], T["abs_lvl_d"], T["abs_dep_d"] = np.abs(T["log_k"]), np.abs(T["lvl_d"]), np.abs(T["dep_d"])
for c in ("decoy", "vburst", "drift", "trend", "garch"): T[c] = H[c].to_numpy(int)
T["ntype"] = H.ntype.to_numpy(int); T["lvl_late"] = (H.lvl_mode == 2).to_numpy(int); T["lvl_icp"] = (H.lvl_mode == 1).to_numpy(int)
va = np.arange(len(XS)) >= int(.85 * len(XS)); tr = ~va
SW = None
if os.environ.get("SA_WEIGHT"):   # realism weights (PRNPE-style focus): label-free discriminator real (train+test) vs these fakes
    from sklearn.model_selection import KFold
    Rr = XR.iloc[:len(lab) + len(sub)][cols].to_numpy(np.float32); Fs = XS.to_numpy(np.float32); SW = np.zeros(len(Fs))
    for a_, b_ in KFold(2, shuffle=True, random_state=0).split(Fs):
        d = lgb.LGBMClassifier(n_estimators=150, learning_rate=.08, num_leaves=31, colsample_bytree=.3, subsample=.8, subsample_freq=1,
                               min_child_samples=50, verbose=-1, n_jobs=int(os.environ.get("SA_JOBS", -1))).fit(np.r_[Rr, Fs[a_]], np.r_[np.ones(len(Rr)), np.zeros(len(a_))])
        p = d.predict_proba(Fs[b_])[:, 1]; SW[b_] = p / (1 - p + 1e-6) * len(a_) / len(Rr)
    SW = np.clip(SW, 0, 5); SW /= SW.mean()
    print(f"realism weights q10/50/90 {np.quantile(SW, [.1, .5, .9]).round(2).tolist()}  {time.time() - t0:.0f}s", flush=True)
fw = (lambda: {} ) if SW is None else (lambda: {"sample_weight": SW[tr]})
out = pd.DataFrame(index=XR.index)
base = dict(learning_rate=.1, num_leaves=31, min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.5, n_estimators=1500, verbose=-1, n_jobs=int(__import__("os").environ.get("SA_JOBS", -1)))
cb = [lgb.early_stopping(40, verbose=False)]
for name, y in T.items():
    if name in ("kind", "family", "ntype"):
        m = lgb.LGBMClassifier(objective="multiclass", **base).fit(XS[tr], y[tr], **fw(), eval_set=[(XS[va], y[va])], callbacks=cb)
        pr = m.predict_proba(XR); labs = KINDS if name == "kind" else FAMI if name == "family" else ["gauss", "t", "bimodal"]
        for i, l in enumerate(labs): out[f"saf_{name}_{l}"] = pr[:, i]
        print(f"  {name:10s} val accuracy {(m.predict_proba(XS[va]).argmax(1) == y[va]).mean():.3f}  trees {m.best_iteration_}  {time.time() - t0:.0f}s", flush=True)
    elif y.dtype.kind == "i":
        m = lgb.LGBMClassifier(**base).fit(XS[tr], y[tr], **fw(), eval_set=[(XS[va], y[va])], callbacks=cb)
        out[f"saf_{name}"] = m.predict_proba(XR)[:, 1]
        print(f"  {name:10s} val AUC {roc_auc_score(y[va], m.predict_proba(XS[va])[:, 1]):.3f}  trees {m.best_iteration_}  {time.time() - t0:.0f}s", flush=True)
    else:
        m = lgb.LGBMRegressor(**base).fit(XS[tr], y[tr], **fw(), eval_set=[(XS[va], y[va])], callbacks=cb)
        out[f"saf_{name}"] = m.predict(XR)
        print(f"  {name:10s} val R2 {r2_score(y[va], m.predict(XS[va])):.3f}  trees {m.best_iteration_}  {time.time() - t0:.0f}s", flush=True)
out.to_parquet(OUT)
print(f"done {out.shape} {time.time() - t0:.0f}s")
