"""Production confirm WITH fold-clean pseudo-labels (2026-11-07 sprint; quick test +.0019 mean over 4 seeds).
Per fold and seed: inner split a/b of the fold-train part (as train.fit_lgb). Teacher = LGB on a (production params), early stop on b.
Teacher test predictions (seed-averaged) -> test rows in the top / bottom Q get hard labels 1 / 0. Student = LGB on a + pseudo rows
(weight PW), early stop on b (real rows only). OOF from the val fold, test predictions averaged over folds x seeds.
    PYTHONPATH=. python probe/confirm_pseudo.py FAMS... --pat PAT --tag TAG [--seeds 3] [--q .25] [--pw 1]
Saves outputs/factory/confirm_{TAG}.npz (student oof/test) and confirm_{TAG}_teacher.npz."""
import argparse, json, os, time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from train import LGB_PARAMS

ap = argparse.ArgumentParser()
ap.add_argument("fams", nargs="*"); ap.add_argument("--pat", default=None); ap.add_argument("--tag", required=True)
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--q", type=float, default=.25); ap.add_argument("--pw", type=float, default=1.0)
ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--lr", type=float, default=None)
args = ap.parse_args()
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str}); y = lab.to_numpy()
ld = lambda p, idx: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
keep = json.load(open("outputs/harvest_keep.json"))
def frame(idx, part):
    N = pd.concat([ld(f"outputs/factory/{f}.parquet", idx) for f in args.fams], axis=1) if args.fams else pd.DataFrame(index=idx)
    if args.pat: N = N.loc[:, N.columns.str.contains(args.pat, regex=True)]
    return pd.concat([ld(f"outputs/feat_{part}.parquet", idx), ld(f"outputs/magic_b_{part}.parquet", idx),
                      ld(f"outputs/harvest_{part}.parquet", idx)[keep], N.loc[:, ~N.columns.duplicated()]], axis=1)
X, Xt = frame(lab.index, "train"), frame(sub["id"], "test"); Xt = Xt[X.columns]
print(f"{args.tag}: {X.shape[1]} columns, q {args.q} pw {args.pw} seeds {args.seeds}", flush=True)
P = {**LGB_PARAMS, "n_jobs": int(os.environ.get("CONF_JOBS", 30)), **({"learning_rate": args.lr} if args.lr else {})}
Xn, Xtn = X.to_numpy(np.float32), Xt.to_numpy(np.float32)
def fit(Xa, ya, Xb, yb, seed, w=None):
    m = lgb.LGBMClassifier(**P, random_state=seed)
    m.fit(Xa, ya, sample_weight=w, eval_set=[(Xb, yb)], eval_metric="auc", callbacks=[lgb.early_stopping(300, verbose=False)]); return m
t0 = time.time(); folds = [int(f) for f in args.folds.split(",")]
oT, oS = np.full(len(y), np.nan), np.full(len(y), np.nan); pT, pS = np.zeros(len(Xt)), np.zeros(len(Xt))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xn, y)):
    if k not in folds: continue
    splits = [train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=100 * k + s) for s in range(args.seeds)]
    teach, tp = [], np.zeros(len(Xt))
    for s, (a, b) in enumerate(splits):
        m = fit(Xn[a], y[a], Xn[b], y[b], 100 * k + s); teach.append(m); tp += m.predict_proba(Xtn)[:, 1] / args.seeds
    oT[va] = np.mean([m.predict_proba(Xn[va])[:, 1] for m in teach], 0); pT += tp / len(folds)
    lo, hi = np.quantile(tp, args.q), np.quantile(tp, 1 - args.q); sel = (tp <= lo) | (tp >= hi); yp = (tp >= hi).astype(int)[sel]
    stu = []
    for s, (a, b) in enumerate(splits):
        m = fit(np.vstack([Xn[a], Xtn[sel]]), np.r_[y[a], yp], Xn[b], y[b], 100 * k + s, np.r_[np.ones(len(a)), args.pw * np.ones(sel.sum())])
        stu.append(m); pS += m.predict_proba(Xtn)[:, 1] / (args.seeds * len(folds))
    oS[va] = np.mean([m.predict_proba(Xn[va])[:, 1] for m in stu], 0)
    print(f"fold {k}: teacher {roc_auc_score(y[va], oT[va]):.4f}  student {roc_auc_score(y[va], oS[va]):.4f}  pseudo {sel.sum()} rows  {time.time()-t0:.0f}s", flush=True)
ok = ~np.isnan(oS)
rng = np.random.default_rng(0); idx = np.where(ok)[0]
d = [roc_auc_score(y[i], oS[i]) - roc_auc_score(y[i], oT[i]) for i in (rng.choice(idx, len(idx)) for _ in range(300))]
print(f"OOF teacher {roc_auc_score(y[ok], oT[ok]):.4f}  student {roc_auc_score(y[ok], oS[ok]):.4f}  gain CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  {time.time()-t0:.0f}s")
if ok.all():
    np.savez(f"outputs/factory/confirm_{args.tag}.npz", oof=oS, test=pS); np.savez(f"outputs/factory/confirm_{args.tag}_teacher.npz", oof=oT, test=pT)
    print("saved")
