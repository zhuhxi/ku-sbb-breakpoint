"""Pairwise interactions of the top-K gain features (fold-0 quick LGB importance): differences and ratios of ranks
(rank-transformed so scale-free), added to the base; quick 5-fold vs base. K from argv."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, itertools
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import rankdata
sys.path.insert(0, "probe"); import feats
K = int(sys.argv[1]) if len(sys.argv) > 1 else 30; t0 = time.time()
X, Xt, y = feats.load(); cols = X.columns; Xn = X.to_numpy(np.float32)
P = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=float(os.environ.get("CS", .5)), reg_lambda=1.0, n_estimators=2000, verbose=-1, n_jobs=int(os.environ.get("J", 64)))
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xn, y))
XT = Xt.to_numpy(np.float32); ALL = np.vstack([Xn, XT]); ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL)
RK = lambda j: (rankdata(ALL[:, j]) / len(ALL)).astype(np.float32)
pairs = list(itertools.combinations(range(K), 2))
FA = pd.concat([pd.read_parquet("outputs/factory/semiauto_bag6.parquet").reindex(feats.lab.index), pd.read_parquet("outputs/factory/semiauto_bag6.parquet").reindex(feats.sub["id"])])
FAM = FA[[c for c in FA if c.startswith("saf_family_")]].to_numpy().argmax(1)
CTXc = [c for c in FA if c.startswith("saf_family_")] + ["saf_kind_None", "saf_garch", "saf_trend"]
CTX = np.column_stack([rankdata(FA[c].to_numpy()) / len(FA) for c in CTXc]).astype(np.float32)
def ctxprod(top):
    R = np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in top]).astype(np.float32)
    return np.column_stack([R[:, i] * CTX[:, c] for i in range(len(top)) for c in range(CTX.shape[1])]).astype(np.float32)
def trip(top):
    T = int(os.environ.get("T", 20)); R = np.column_stack([rankdata(ALL[:, j]) / len(ALL) for j in top[:T]]).astype(np.float32)
    P2 = [R[:, i] - R[:, j] for i, j in pairs] + [R[:, i] * R[:, j] for i, j in pairs] if os.environ.get("WITHPAIRS") else []
    if P2: P2 = [np.column_stack([rankdata(ALL[:, top[i]]) / len(ALL) - rankdata(ALL[:, top[j]]) / len(ALL) for i, j in pairs] + [rankdata(ALL[:, top[i]]) / len(ALL) * rankdata(ALL[:, top[j]]) / len(ALL) for i, j in pairs]).astype(np.float32)]
    return np.hstack(P2 + [np.column_stack([R[:, i] * R[:, j] * R[:, k] for i, j, k in itertools.combinations(range(T), 3)]).astype(np.float32)])
SEG = None
def lnorm(top):
    global SEG
    if SEG is None:
        Sx = pd.concat([pd.read_parquet("data/train_series.parquet"), pd.read_parquet("data/test_series.parquet")]); Sx["id"] = Sx.id.astype(str)
        g = Sx.groupby("id"); n = g.size(); n0 = g.period.apply(lambda v: (v == 0).sum()); seg = np.minimum(n0, n - n0)
        ids = list(feats.lab.index) + list(feats.sub["id"]); SEG = pd.qcut(seg.reindex(ids), 10, labels=False).to_numpy()
    out = np.zeros((len(ALL), len(top)), np.float32)
    for b in range(10):
        m = SEG == b
        for c, j in enumerate(top):
            v = ALL[m, j]; q1, q2, q3 = np.percentile(v, [25, 50, 75]); out[m, c] = (v - q2) / (q3 - q1 + 1e-9)
    return out
def famrank(top):
    out = np.zeros((len(ALL), len(top)), np.float32)
    for f in np.unique(FAM):
        m = FAM == f
        for c, j in enumerate(top): out[m, c] = rankdata(ALL[m, j]) / m.sum()
    return out
oof = {"base": np.zeros(len(y)), "new": np.zeros(len(y))}; te = np.zeros(len(XT))
for k, (tr, va) in enumerate(folds):
    a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**P, random_state=int(os.environ.get("SEED",0))).fit(Xn[a], y[a], eval_set=[(Xn[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof["base"][va] = m0.predict_proba(Xn[va])[:, 1]
    top = np.argsort(-m0.booster_.feature_importance("gain"))[:K]; R = np.column_stack([RK(j) for j in top])
    MODE = os.environ.get("MODE", "both"); N = np.column_stack([R[:, i] - R[:, j] for i, j in pairs] + [R[:, i] * R[:, j] for i, j in pairs] + [(R[:, i] + .01) / (R[:, j] + .01) for i, j in pairs] + [R[:, i] * R[:, j] ** 2 for i, j in pairs]).astype(np.float32) if MODE == "ext" else trip(top) if MODE == "trip" else lnorm(top) if MODE == "lnorm" else np.hstack([ctxprod(top), np.column_stack([R[:, i] - R[:, j] for i, j in pairs] + [R[:, i] * R[:, j] for i, j in pairs])]) if MODE == "all" else ctxprod(top) if MODE == "ctx" else famrank(top) if MODE == "fam" else np.column_stack(([R[:, i] - R[:, j] for i, j in pairs] if MODE != "prod" else []) + ([R[:, i] * R[:, j] for i, j in pairs] if MODE != "diff" else [])).astype(np.float32)
    Z = np.hstack([ALL, N]); Ztr, Zte = Z[:len(y)], Z[len(y):]
    if os.environ.get("TWO"):   # second stage: top-K of (base + pairs) by a quick model on the inner-train part, pairs among those
        m1 = lgb.LGBMClassifier(**P, random_state=0).fit(Ztr[a], y[a], eval_set=[(Ztr[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
        t2 = np.argsort(-m1.booster_.feature_importance("gain"))[:K]; R2 = np.column_stack([rankdata(Z[:, j]) / len(Z) for j in t2]).astype(np.float32)
        N2 = np.column_stack([R2[:, i] - R2[:, j] for i, j in pairs] + [R2[:, i] * R2[:, j] for i, j in pairs]).astype(np.float32)
        Z = np.hstack([Z, N2]); Ztr, Zte = Z[:len(y)], Z[len(y):]
    mm = lgb.LGBMClassifier(**P, random_state=int(os.environ.get("SEED",0))).fit(Ztr[a], y[a], eval_set=[(Ztr[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof["new"][va] = mm.predict_proba(Ztr[va])[:, 1]; te += mm.predict_proba(Zte)[:, 1] / 5
    print(f"fold {k} base {roc_auc_score(y[va], oof['base'][va]):.4f} new {roc_auc_score(y[va], oof['new'][va]):.4f} {time.time()-t0:.0f}s", flush=True)
np.save(f"outputs/arms/pair{K}_oof.npy", oof["new"]); np.save(f"outputs/arms/pair{K}_test.npy", te)
rng = np.random.default_rng(0); d = [roc_auc_score(y[i], oof["new"][i]) - roc_auc_score(y[i], oof["base"][i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(200))]
print(f"K {K}: base {roc_auc_score(y, oof['base']):.4f} new {roc_auc_score(y, oof['new']):.4f} gain {roc_auc_score(y, oof['new'])-roc_auc_score(y, oof['base']):+.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] {time.time()-t0:.0f}s")
