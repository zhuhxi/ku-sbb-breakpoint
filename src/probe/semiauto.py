"""Semi-automatic ABC (Fearnhead & Prangle 2012; ABCpy Semiautomatic) with our own models: on best2 fakes, one LightGBM
per HIDDEN generator parameter (features -> parameter; held-out simulated validation set for early stopping, as in
ABCpy SemiautomaticNN), then the fitted values on the REAL series are new features.
Targets: kind (none / spread / level / dep / shape), family (6), log spread ratio, level shift (pre noise sd units),
AR coefficient change, p1, p2, decoy / vburst / drift / trend flags.  Features: the 119 fast features (as probe/sim_post.py).
    python probe/semiauto.py NFAKE [OUT]  -> OUT (default outputs/factory/semiauto.parquet)
v2 targets (OUT given): + noise type (Gaussian / t / bimodal), 1/df, GARCH, sine amplitude, delayed / intercept level mode.
"""
import importlib.util, json, sys, time
import lightgbm as lgb, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, roc_auc_score
sys.path.insert(0, "probe")
from ku_gen import DEFAULT, gen

FAMS = ["joint_lr", "kind_lr", "scan_lr", "persist_feats", "mosum_feats", "ramp_step", "target_feats", "decoy_feats", "scan_cp"]
_M = {}


def feats(x, n0):
    if not _M:
        for f in FAMS:
            sp = importlib.util.spec_from_file_location(f, f"probe/{f}.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); _M[f] = m
    out = {}
    for f in FAMS:
        try: out.update(_M[f].one(x, n0))
        except Exception: pass
    return out


B = {k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v) for k, v in json.load(open("outputs/factory/ku_gen_best2.json")).items()}
P = dict(DEFAULT, **B)
KINDS, FAMI = [None, "spread", "level", "dep", "shape"], ["WN", "AR1", "AR2", "SIN", "RW", "LL"]


def fake(s):
    x, n0, h = gen(P, 7_000_000 + s, bool(s % 2))
    return feats(x, n0), h


if __name__ == "__main__":
    import semiauto as SA
    t0 = time.time(); NF = int(sys.argv[1]); OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/factory/semiauto.parquet"
    R = Parallel(n_jobs=30, batch_size=32)(delayed(SA.fake)(s) for s in range(NF))
    XR = pd.read_parquet("outputs/factory/fastfam_real.parquet")
    XF = pd.DataFrame([r[0] for r in R]).astype(np.float32).reindex(columns=XR.columns)
    H = pd.DataFrame([r[1] for r in R])
    print(f"fakes {XF.shape} {time.time() - t0:.0f}s", flush=True)
    T = {"kind": H.kind.map({k: i for i, k in enumerate(KINDS)}).to_numpy(), "family": H.family.map({k: i for i, k in enumerate(FAMI)}).to_numpy()}
    for c in ("log_k", "lvl_d", "dep_d", "p1", "p2"): T[c] = H[c].to_numpy(float)
    T["abs_log_k"], T["abs_lvl_d"], T["abs_dep_d"] = np.abs(T["log_k"]), np.abs(T["lvl_d"]), np.abs(T["dep_d"])
    for c in ("decoy", "vburst", "drift", "trend"): T[c] = H[c].to_numpy(int)
    if len(sys.argv) > 2:
        T["ntype"] = H.ntype.to_numpy(int); T["inv_df"] = H.inv_df.to_numpy(float); T["garch"] = H.garch.to_numpy(int)
        T["sin_amp"] = H.sin_amp.to_numpy(float); T["lvl_late"] = (H.lvl_mode == 2).to_numpy(int); T["lvl_icp"] = (H.lvl_mode == 1).to_numpy(int)
    va = np.arange(NF) >= int(.85 * NF); tr = ~va
    out = pd.DataFrame(index=XR.index)
    base = dict(learning_rate=.05, num_leaves=31, min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.7, n_estimators=2000, verbose=-1, n_jobs=30)
    cb = [lgb.early_stopping(50, verbose=False)]
    for name, y in T.items():
        if name in ("kind", "family", "ntype"):
            m = lgb.LGBMClassifier(objective="multiclass", **base).fit(XF[tr], y[tr], eval_set=[(XF[va], y[va])], callbacks=cb)
            pr = m.predict_proba(XR); pv = m.predict_proba(XF[va])
            labs = KINDS if name == "kind" else FAMI if name == "family" else ["gauss", "t", "bimodal"]
            for i, l in enumerate(labs): out[f"sa_{name}_{l}"] = pr[:, i]
            print(f"  {name:10s} val accuracy {(pv.argmax(1) == y[va]).mean():.3f}  trees {m.best_iteration_}", flush=True)
        elif name in ("decoy", "vburst", "drift", "trend", "garch", "lvl_late", "lvl_icp"):
            m = lgb.LGBMClassifier(**base).fit(XF[tr], y[tr], eval_set=[(XF[va], y[va])], callbacks=cb)
            out[f"sa_{name}"] = m.predict_proba(XR)[:, 1]
            print(f"  {name:10s} val AUC {roc_auc_score(y[va], m.predict_proba(XF[va])[:, 1]):.3f}  trees {m.best_iteration_}", flush=True)
        else:
            m = lgb.LGBMRegressor(**base).fit(XF[tr], y[tr], eval_set=[(XF[va], y[va])], callbacks=cb)
            out[f"sa_{name}"] = m.predict(XR)
            print(f"  {name:10s} val R2 {r2_score(y[va], m.predict(XF[va])):.3f}  trees {m.best_iteration_}", flush=True)
    out.to_parquet(OUT)
    print(f"done {out.shape} {time.time() - t0:.0f}s")
