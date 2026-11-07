"""Generator-trace scan for KU-SBB: look for label signal in things that should carry none.

    python forensics.py            # prints every check; anything with |AUC-0.5| > 0.03 is flagged

Each block is one family of checks. Univariate AUC is folded to >= 0.5 (direction ignored).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import lightgbm as lgb

D = Path("data")
tr = pd.read_parquet(D / "train_series.parquet")
te = pd.read_parquet(D / "test_series.parquet")
lab = pd.read_csv(D / "train_labels.csv", dtype={"id": str})
y = lab.set_index("id")["label"]
FLAG = 0.03


def auc(v, yy=None):
    yy = y.loc[v.index] if yy is None else yy
    v = v.replace([np.inf, -np.inf], np.nan)
    v = v.fillna(v.median())
    if v.nunique() < 2:
        return 0.5
    a = roc_auc_score(yy, v)
    return max(a, 1 - a)


def report(title, table):
    print(f"\n== {title}")
    for k, v in table.items():
        a = auc(v) if isinstance(v, pd.Series) else v
        print(f"  {'!!' if abs(a - 0.5) > FLAG else '  '} {k:28s} AUC {a:.3f}")


def per_series(df):
    out = {}
    for sid, g in df.groupby("id", sort=False):
        v, p, t = g["value"].to_numpy(), g["period"].to_numpy(), g["time"].to_numpy()
        n0 = int((p == 0).sum())
        x0, x1 = v[:n0], v[n0:]
        d = np.diff(v)
        sd_d = np.median(np.abs(d)) * 1.4826 + 1e-12
        sd0 = np.median(np.abs(x0 - np.median(x0))) * 1.4826 + 1e-12
        # float representation
        f32_exact = np.mean(v.astype(np.float32).astype(np.float64) == v)
        dec = np.array([len(repr(float(a)).split(".")[-1].rstrip("0")) if "e" not in repr(float(a)) else 17 for a in v[:200]])
        out[sid] = {
            "n": len(v), "n0": n0, "n1": len(v) - n0, "frac_b": n0 / len(v),
            "n_mod2": len(v) % 2, "n_mod8": len(v) % 8, "n0_mod10": n0 % 10,
            "frac_b_round2": abs(n0 / len(v) - round(n0 / len(v), 2)),
            "pow2_len": float(len(v) & (len(v) - 1) == 0),
            "time_ok": float(t[0] == 0 and np.all(np.diff(t) == 1)),
            "period_sorted": float(np.all(np.diff(p) >= 0)),
            "f32_exact": f32_exact, "decimals_mean": dec.mean(),
            "dup_values": 1 - len(np.unique(v)) / len(v),
            "round_values": np.mean(np.abs(v - np.round(v, 2)) < 1e-12),
            "first_value": v[0] / sd0, "last_value": (v[-1] - np.median(x0)) / sd0,
            "first_abs": abs(v[0] - np.median(x0)) / sd0,
            # seams at the boundary: is the after-part stitched on?
            "seam_jump": abs(v[n0] - v[n0 - 1]) / sd_d,
            "seam_jump_signed": (v[n0] - v[n0 - 1]) / sd_d,
            "seam_vs_max_diff": abs(v[n0] - v[n0 - 1]) / (np.abs(d).max() + 1e-12),
            "seam_rank": np.mean(np.abs(d) < abs(v[n0] - v[n0 - 1])),
            "seam_first_after_dev": abs(v[n0] - x1[:20].mean()) / sd0,
            "seam_restart_to_zero": abs(v[n0]) / sd0,
            "seam_mean_of_5": abs(x1[:5].mean() - x0[-5:].mean()) / sd0,
            # global (non-contrast) statistics
            "g_mean": v.mean(), "g_std": v.std(), "g_cv": v.std() / (abs(v.mean()) + 1e-12),
            "g_absmax_over_std": np.abs(v - v.mean()).max() / (v.std() + 1e-12),
            "g_mean_is_0": abs(v.mean()) / (v.std() + 1e-12), "g_std_near_1": abs(np.log(v.std() + 1e-12)),
            "b_mean_raw": x0.mean(), "b_std_raw": x0.std(),
            "start_sum_hash": int(hashlib.md5(np.round(v[:10], 6).tobytes()).hexdigest()[:8], 16),
        }
    return pd.DataFrame.from_dict(out, orient="index")


S = per_series(tr).loc[y.index]

report("lengths and boundary placement", {k: S[k] for k in ("n", "n0", "n1", "frac_b", "n_mod2", "n_mod8", "n0_mod10", "frac_b_round2", "pow2_len")})
report("time / period bookkeeping", {k: S[k] for k in ("time_ok", "period_sorted")})
report("numeric representation", {k: S[k] for k in ("f32_exact", "decimals_mean", "dup_values", "round_values")})
report("seams at the boundary (after-part stitched on?)", {k: S[k] for k in S.columns if k.startswith("seam")})
report("start / end values", {k: S[k] for k in ("first_value", "first_abs", "last_value")})
report("global statistics (no before/after contrast)", {k: S[k] for k in S.columns if k.startswith(("g_", "b_"))})

# ---- ordering and identifiers ----
print("\n== ordering and identifiers")
order_train = pd.Series(np.arange(len(lab)), index=lab["id"])
print(f"   label-file row order AUC {auc(order_train):.3f}")
first_row = tr.drop_duplicates("id").reset_index()[["id", "index"]].set_index("id")["index"]
print(f"   series-file row order AUC {auc(first_row.loc[y.index]):.3f}")
lab_seq = lab["label"].to_numpy()
print(f"   P(label[i]==label[i+1]) = {np.mean(lab_seq[1:] == lab_seq[:-1]):.3f} (0.5 if shuffled)")
idnum = pd.Series([int(s[:8], 16) for s in y.index], index=y.index)
print(f"   id hex prefix AUC {auc(idnum):.3f}   id first hex digit pos-rate spread "
      f"{y.groupby([s[0] for s in y.index]).mean().agg(['min', 'max']).round(3).tolist()}")
for name, fn in (("md5(i)", lambda i: hashlib.md5(str(i).encode()).hexdigest()),
                 ("md5(train_i)", lambda i: hashlib.md5(f"train_{i}".encode()).hexdigest())):
    hits = len(set(fn(i) for i in range(30000)) & set(y.index))
    print(f"   ids equal to {name} for i<30000: {hits}")

# ---- duplicates and reused noise ----
print("\n== duplicated / copied material")
T = per_series(te)
for nm, A in (("train", S), ("test", T)):
    print(f"   {nm}: series sharing their first-10-values hash with another series: {A['start_sum_hash'].duplicated(keep=False).sum()}")
print(f"   train/test sharing first-10-values hash: {len(set(S['start_sum_hash']) & set(T['start_sum_hash']))}")
# reused innovations: correlation of first 64 diffs between random pairs vs nearest neighbour
Z = []
for sid, g in tr.groupby("id", sort=False):
    d = np.diff(g["value"].to_numpy()[:65])
    Z.append((d - d.mean()) / (d.std() + 1e-12))
Z = np.array(Z)[:6000]
C = (Z @ Z.T) / 64
np.fill_diagonal(C, 0)
print(f"   max |corr| of first 64 diffs among 6000 train series: {np.abs(C).max():.3f} (random ~0.6 at this scale)")

# ---- train vs test shift (adversarial validation) ----
print("\n== adversarial validation (train vs test on the same scan columns)")
cols = [c for c in S.columns if c != "start_sum_hash"]
A = pd.concat([S[cols], T[cols]])
is_test = np.r_[np.zeros(len(S)), np.ones(len(T))]
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, verbose=-1), A, is_test,
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"   train-vs-test AUC {roc_auc_score(is_test, p):.3f} (0.5 = same distribution)")

# ---- all scan columns together ----
print("\n== everything above in one model (should be ~0.5 if there is no trace)")
cols = [c for c in S.columns if not c.startswith(("seam", "start_sum"))]
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, verbose=-1), S[cols], y.loc[S.index],
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"   non-seam trace columns -> label AUC {roc_auc_score(y.loc[S.index], p):.3f}")
S.to_parquet("outputs/forensics_train.parquet")
T.to_parquet("outputs/forensics_test.parquet")

# ---- the decisive test: can the BEFORE segment alone predict the label? ----
# A clean generator draws the break independently of the pre-break process, so this must be ~0.5.
# Anything clearly above means process "families" with different break rates: a usable trace.
feat = Path("outputs/feat_train.parquet")
if feat.exists():
    F = pd.read_parquet(feat).loc[y.index]
    bcols = [c for c in F.columns if c.startswith("b_") or (c.endswith("_b") and c != "spike_a")] + ["n0", "n1"]
    p = cross_val_predict(lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, verbose=-1), F[bcols], y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print(f"\n== before-segment only ({len(bcols)} cols) -> label AUC {roc_auc_score(y, p):.3f}")


# ---- reused noise: high raw correlation is only a trace if it survives AR whitening ----
def _whiten(x, p=8):
    X = np.column_stack([x[p - k - 1:len(x) - k - 1] for k in range(p)] + [np.ones(len(x) - p)])
    b, *_ = np.linalg.lstsq(X, x[p:], rcond=None)
    return x[p:] - X @ b


vals = [g["value"].to_numpy() for df in (tr, te) for _, g in df.groupby("id", sort=False)]
L = min(len(v) for v in vals) - 2
Z = np.array([(lambda d: (d - d.mean()) / (d.std() + 1e-12))(np.diff(v[:L + 1])) for v in vals], np.float32)
worst = 0.0
for i in range(0, len(Z), 2000):
    C = np.abs(Z[i:i + 2000] @ Z.T / L)
    C[np.arange(len(C)), i + np.arange(len(C))] = 0
    for k in np.where(C.max(1) > 0.5)[0]:
        a, b = vals[i + k], vals[int(C[k].argmax())]
        m = min(len(a), len(b))
        worst = max(worst, abs(np.corrcoef(_whiten(a[:m]), _whiten(b[:m]))[0, 1]))
print(f"== reused noise: max whitened |corr| among highly correlated pairs {worst:.3f} (near 1 = copied noise)")


# ---- float bit level: the lowest set bit records the size of hidden intermediates ----
# v = a + b is only as precise as the larger of a, b. So each value's lowest set bit (its "grid")
# leaks how big the generator's intermediate numbers were. Two things to check:
#   1) does the grid carry label signal that plain magnitude does not?   (KU-SBB: +0.001 AUC, real)
#   2) is the arithmetic the same in train and test?                       (KU-SBB: no -> do not use)
def bit_table(df):
    b = df["value"].to_numpy().view(np.uint64)
    mant = (b & ((1 << 52) - 1)) | (1 << 52)
    low = mant & (~mant + np.uint64(1))
    ex = ((b >> 52) & 0x7FF).astype(int) - 1023
    return pd.DataFrame({"id": df["id"].to_numpy(), "period": df["period"].to_numpy(), "exp": ex,
                         "par": (b & 1).astype(int), "L": ex - 52 + np.log2(low.astype(float)).astype(int)})


bt, bs = bit_table(tr), bit_table(te)
pe = pd.DataFrame({"train": bt.groupby("exp").par.mean(), "test": bs.groupby("exp").par.mean(),
                   "share": bt.exp.value_counts(normalize=True)})
print("\n== float LSB parity by binary exponent (same exponent, different parity = different arithmetic)")
print(pe[pe.share > 0.01].round(3).to_string())
Gt = bt.groupby(["id", "period"]).L.min().unstack()
Gt = (Gt[1] - Gt[0]).loc[y.index]
print(f"   grid shift after-vs-before (Lmin_a - Lmin_b) -> label AUC {auc(Gt):.3f}")
gb = lambda t: t.groupby(["id", "period"]).L.min().unstack()
Z = pd.concat([gb(bt), gb(bs)])
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, verbose=-1), Z, np.r_[np.zeros(len(S)), np.ones(len(T))],
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"   train-vs-test AUC on grid features {roc_auc_score(np.r_[np.zeros(len(S)), np.ones(len(T))], p):.3f} (0.5 = transferable)")


# ---- family marker: series whose final values were multiplied by a non-power-of-two ----
# In x_t = phi*x_{t-1} + e_t, whenever |x_{t-1}|/|x_t| > 8 the result must carry >= 3 extra zero
# bits (cancellation). A final rescaling wipes them: trailing zeros fall to ~1 (geometric).
# KU-SBB: 1.7% of train vs 18% of test, and those test series are a drifted population.
def rescaled_marker(t, df):
    lr = np.log2(np.abs(df.groupby("id")["value"].shift(1).to_numpy()) / (np.abs(df["value"].to_numpy()) + 1e-300))
    tz = t["L"].to_numpy() - (t["exp"].to_numpy() - 52)
    m = pd.DataFrame({"id": t["id"], "tz": tz})[lr > 3].groupby("id").tz.mean()
    return m < 1.5


mt, ms = rescaled_marker(bt, tr).reindex(S.index).fillna(False), rescaled_marker(bs, te).reindex(T.index).fillna(False)
print(f"\n== rescaled-family marker: train {mt.mean():.2%}  test {ms.mean():.2%}  (train label rate in family {y.loc[mt[mt].index].mean():.3f})")
fcols = pd.read_parquet(feat).columns if feat.exists() else None
if fcols is not None and Path("outputs/feat_test.parquet").exists():
    Ftr = pd.read_parquet(feat).loc[S.index].replace([np.inf, -np.inf], np.nan)
    Fte = pd.read_parquet("outputs/feat_test.parquet").loc[T.index].replace([np.inf, -np.inf], np.nan)
    for name, a, b in (("train normal vs test normal", Ftr[~mt.values], Fte[~ms.values]),
                       ("train normal vs test family", Ftr[~mt.values], Fte[ms.values])):
        Z = pd.concat([a, b]); t_ = np.r_[np.zeros(len(a)), np.ones(len(b))]
        p = cross_val_predict(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, verbose=-1), Z, t_,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        print(f"   adversarial on ordinary features, {name}: AUC {roc_auc_score(t_, p):.3f}")
