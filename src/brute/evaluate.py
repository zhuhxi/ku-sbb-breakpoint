"""Full brute-force interaction search, step 3: re-rank survivors, judge on the holdout only.

    python brute/evaluate.py [--swap]

Survivors (above the noise ceiling, capped at the 2000 best) are rebuilt from their formulas,
re-ranked by LightGBM gain on screening rows, and the top 200 kept. The verdict is 5-fold CV
inside the holdout, 5 seeds, paired bootstrap: base vs base+pairs vs base+pairs+triples.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import LGB_PARAMS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--swap", action="store_true")
ap.add_argument("--cap", type=int, default=2000)
ap.add_argument("--keep", type=int, default=200)
ap.add_argument("--ignore_null", action="store_true", help="diagnostic: take top-scoring pairs even below the noise ceiling")
args = ap.parse_args()
out = Path("outputs/brute") / ("swap" if args.swap else "main")
z = np.load(out / "prep.npz", allow_pickle=True)
y, scr, hold, n_base = z["y"], z["screen"], z["hold"], int(z["n_base"])
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
base = pd.read_parquet("outputs/feat_train.parquet").loc[lab.index].replace([np.inf, -np.inf], np.nan)
EPS = np.median(np.abs(z["raw"][scr]), axis=0) * 1e-3 + 1e-9


def build(raw, rank, rows):
    out_ = {}
    for r in rows.itertuples():
        a, b = int(r.a), int(r.b)
        A, B, ra, rb = raw[:, a].astype(np.float64), raw[:, b].astype(np.float64), rank[:, a], rank[:, b]
        if hasattr(r, "c") and not pd.isna(getattr(r, "c", np.nan)):
            c = int(r.c)
            Cc, rc = raw[:, c].astype(np.float64), rank[:, c]
            v = {"mul3": lambda: A * B * Cc, "muldiv": lambda: A * B / (np.abs(Cc) + EPS[c]) * np.where(Cc >= 0, 1, -1),
                 "r_addmul": lambda: (ra + rb) * rc, "r_submul": lambda: (ra - rb) * rc}[r.op]()
            name = f"{r.op}__{a}_{b}_{c}"
        else:
            sb, sa = np.where(B >= 0, 1, -1), np.where(A >= 0, 1, -1)
            v = {"mul": lambda: A * B, "div": lambda: A / (np.abs(B) + EPS[b]) * sb, "rdiv": lambda: B / (np.abs(A) + EPS[a]) * sa,
                 "mul_b2": lambda: A * B * B, "mul_a2": lambda: A * A * B,
                 "logratio": lambda: np.log(np.abs(A) + EPS[a]) - np.log(np.abs(B) + EPS[b]),
                 "r_add": lambda: ra + rb, "r_sub": lambda: ra - rb, "r_mul": lambda: ra * rb, "r_max": lambda: np.maximum(ra, rb),
                 "r_min": lambda: np.minimum(ra, rb), "r_absdiff": lambda: np.abs(ra - rb)}[r.op]()
            name = f"{r.op}__{a}_{b}"
        out_[name] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return pd.DataFrame(out_)


P = pd.read_parquet(out / "screen_pairs.parquet")
P = (P if args.ignore_null else P[P.passed]).head(args.cap)
tri_path = out / "screen_triples.parquet"
T = pd.read_parquet(tri_path) if tri_path.exists() else pd.DataFrame(columns=["op", "a", "b", "c", "score", "passed"])
T = T[T.passed].head(args.cap) if len(T) else T
print(f"survivors: pairs {len(P)}, triples {len(T)}")

CP = build(z["raw"], z["rank"], P).set_index(lab.index) if len(P) else pd.DataFrame(index=lab.index)
CT = build(z["raw"], z["rank"], T).set_index(lab.index) if len(T) else pd.DataFrame(index=lab.index)


def rerank(C, seed):
    if C.shape[1] == 0:
        return []
    X = pd.concat([base, C], axis=1).iloc[scr]
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 600, "learning_rate": 0.05, "colsample_bytree": 0.3},
                           random_state=seed).fit(X, y[scr])
    g = pd.Series(m.booster_.feature_importance("gain"), X.columns)
    return g[C.columns].nlargest(args.keep).index.tolist()


sel_p = rerank(CP, 0)
sel_all = rerank(pd.concat([CP, CT], axis=1), 1)
arms = {"base": base, "base+pairs": pd.concat([base, CP[sel_p]], axis=1),
        "base+pairs+triples": pd.concat([base, CP.reindex(columns=[c for c in sel_all if c in CP]),
                                         CT.reindex(columns=[c for c in sel_all if c in CT])], axis=1)}

yh = y[hold]
oof = {k: np.zeros(len(hold)) for k in arms}
for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=3).split(hold, yh)):
    for s in range(5):
        for k, X in arms.items():
            Xh = X.iloc[hold]
            a, b = train_test_split(tr, test_size=0.1, stratify=yh[tr], random_state=100 * f + s)
            m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=100 * f + s)
            m.fit(Xh.iloc[a], yh[a], eval_set=[(Xh.iloc[b], yh[b])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(300, verbose=False)])
            oof[k][va] += m.predict_proba(Xh.iloc[va])[:, 1] / 5
    print(f"holdout fold {f}: " + "  ".join(f"{k} {roc_auc_score(yh[va], oof[k][va]):.4f}" for k in arms), flush=True)

rng = np.random.default_rng(0)
idx = [rng.integers(0, len(yh), len(yh)) for _ in range(500)]
res = {}
for k in arms:
    d = [roc_auc_score(yh[i], oof[k][i]) - roc_auc_score(yh[i], oof["base"][i]) for i in idx]
    res[k] = {"auc": roc_auc_score(yh, oof[k]), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    print(f"{k:20s} holdout AUC {res[k]['auc']:.4f}  vs base CI [{res[k]['ci'][0]:+.4f}, {res[k]['ci'][1]:+.4f}]")
names = list(z["names"])
pretty = lambda c: c.split("__")[0] + "(" + ", ".join(names[int(i)] for i in c.split("__")[1].split("_")) + ")"
json.dump({"result": res, "pairs_selected": [pretty(c) for c in sel_p], "all_selected": [pretty(c) for c in sel_all]},
          open(out / "selected.json", "w"), indent=1)
print("top selected:", [pretty(c) for c in sel_all[:8]])
