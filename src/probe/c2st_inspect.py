"""Which real break series does the generator fail to produce? Train real-vs-generated (menu) on before/after
statistics, then list and plot the real series the classifier is most sure are real (lowest fake probability),
with their per-series measurements.

    python probe/c2st_inspect.py MENU
"""
import importlib.util
import sys
import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")
name = sys.argv[1]
sys.argv = [sys.argv[0], "--n", "1"]
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
c2 = load("c2", "probe/break_c2st.py")
bm = load("bm", "probe/break_measure.py")

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = set(np.random.default_rng(11).choice(lab.index[lab == 1], 2000, False))
RK, real = zip(*[(k, (g["value"].to_numpy(), int((g["period"] == 0).sum()))) for k, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")])
fake = Parallel(n_jobs=30)(delayed(c2.cb.gen)(c2.cb.BREAKS[name], 8 * 10 ** 6 + s, True, True) for s in range(2000))
F = pd.DataFrame(Parallel(n_jobs=30)(delayed(c2.feats)(v, n0) for v, n0 in list(real) + fake))
keep = F.cls.isin(["WN", "AR1", "AR2"]).to_numpy()
y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
X = F.drop(columns="cls")
X.columns = [c.replace("(", "").replace(")", "").replace(">", "gt").replace(":", "") for c in X.columns]
p = np.full(len(y), np.nan)
p[keep] = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=30),
                            X[keep], y[keep], cv=5, method="predict_proba")[:, 1]
pr = pd.Series(p[: len(real)], index=RK).dropna()
M = pd.DataFrame([bm.measure(*real[RK.index(k)]) for k in pr.index], index=pr.index, columns=["level_noise", "spread_x", "dep"])
M["p_fake"] = pr
M["n"], M["frac"] = [len(real[RK.index(k)][0]) for k in pr.index], [real[RK.index(k)][1] / len(real[RK.index(k)][0]) for k in pr.index]
lo = M.sort_values("p_fake").head(40)
print("real series most surely real (generator can't make them): measurement medians vs all real")
print(pd.DataFrame({"most-real 40": lo.drop(columns="p_fake").abs().median(), "all real": M.drop(columns="p_fake").abs().median()}).round(3).to_string())
print(lo.head(12).round(3).to_string())
fig, ax = plt.subplots(8, 1, figsize=(15, 16))
for i, k in enumerate(lo.index[:8]):
    v, n0 = real[RK.index(k)]
    ax[i].plot(v, lw=0.5, color="0.4")
    ax[i].plot(pd.Series(v).rolling(30, center=True).mean(), color="tab:red", lw=1.4)
    ax[i].axvline(n0, color="tab:blue", ls="--")
    r = lo.loc[k]
    ax[i].set_title(f"id={k}  level={r.level_noise:+.2f} noise-sd  spread x{r.spread_x:.2f}  dep {r.dep:+.2f}  p_fake={r.p_fake:.2f}", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig("outputs/probe/c2st_most_real.png", dpi=80)

# ---- co-occurrence and big-level analysis -------------------------------------------------------------------------
G = pd.DataFrame([bm.measure(x, n0) for x, n0 in fake], columns=["level_noise", "spread_x", "dep"])
G["cls"] = F.cls.to_numpy()[len(real):]
G = G[G.cls.isin(["WN", "AR1", "AR2"])]
M["cls"] = F.cls.to_numpy()[: len(real)][[RK.index(k) for k in M.index]]


def flags(T):
    return pd.DataFrame({"L": T.level_noise.abs() > 0.75, "S": (T.spread_x > 1.25) | (T.spread_x < 0.8), "D": T.dep.abs() > 0.25})


def combo(T):
    f = flags(T)
    return f.apply(lambda r: "+".join(c for c in "LSD" if r[c]) or "none", axis=1).value_counts(normalize=True).round(3)


hard = M[M.p_fake < 0.2]
M.to_parquet("outputs/probe/c2st_real_scores.parquet")
print(f"\nhard-to-generate real (p_fake < 0.2): {len(hard)} of {len(M)}")
print(pd.DataFrame({"hard real": combo(hard), "all real": combo(M), "generated": combo(G)}).fillna(0).to_string())
qs = [.5, .75, .9, .95, .99]
print("|level| noise-sd among series with |level| > 0.75:")
for nm, T in [("hard real", hard), ("all real", M), ("generated", G)]:
    z = T.level_noise.abs()
    print(f"  {nm:10s} n={int((z > .75).sum()):4d}  q50/75/90/95/99 {np.round(np.quantile(z[z > .75], qs), 2)}")
print("structure share  hard", hard.cls.value_counts(normalize=True).round(2).to_dict(), " all", M.cls.value_counts(normalize=True).round(2).to_dict())
