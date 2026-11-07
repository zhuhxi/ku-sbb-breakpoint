"""Which generated break series look most fake? Real-vs-generated classifier (menu "L abc40 ns35") on before/after
statistics; take the generated series it is surest are fake, show what the generator did to them (structure, which
changes, level size, noise switch) against all generated series, which statistics sit far outside the real range,
and plot 12 of them next to 4 typical real ones.

    python probe/c2st_fakest.py
"""
import sys
import warnings

sys.argv = [sys.argv[0], "--menu", "L abc40 ns35", "--only-break", "1"]
import importlib.util
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")
s = importlib.util.spec_from_file_location("bc", "probe/break_c2st.py")
bc = importlib.util.module_from_spec(s); s.loader.exec_module(bc)
N = 2000
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
ids = set(np.random.default_rng(11).choice(lab.index[lab == 1], N, False))
real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
out = Parallel(n_jobs=30)(delayed(bc.cb.gen)(bc.MENU, 6 * 10 ** 6 + i, True, "info") for i in range(N))
fake = [(x, n0) for x, n0, _ in out]
info = pd.DataFrame([d for _, _, d in out])
F = pd.DataFrame(Parallel(n_jobs=30)(delayed(bc.feats)(v, n0) for v, n0 in real + fake))
src = np.r_[np.zeros(N), np.ones(N)]
keep = F.cls.isin(["WN", "AR1", "AR2"]).to_numpy()
X = F.drop(columns="cls")
X.columns = [c.replace("(", "").replace(")", "").replace(">", "gt").replace(":", "") for c in X.columns]
p = np.full(len(F), np.nan)
p[keep] = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=30),
                            X[keep], src[keep], cv=5, method="predict_proba")[:, 1]
pf = pd.Series(p[N:])
top = pf.nlargest(100).index                                     # the 100 most obviously fake
info["fakest"] = info.index.isin(top)
print("what the generator did: 100 most-fake vs all generated (%)")
cols = ["noise_switch", "on_level", "on_spread", "on_dep"]
T = pd.DataFrame({"most fake": info.loc[top, cols].mean() * 100, "all generated": info[cols].mean() * 100})
for k in ("WN", "AR1", "AR2"):
    T.loc[f"structure {k}"] = [(info.loc[top, "structure"] == k).mean() * 100, (info.structure == k).mean() * 100]
T.loc["large level jump (>1.5)"] = [(info.loc[top, "level"] > 1.5).mean() * 100, (info.level > 1.5).mean() * 100]
T.loc["medium level jump (.7-1)"] = [info.loc[top, "level"].between(.7, 1).mean() * 100, info.level.between(.7, 1).mean() * 100]
print(T.round(1).to_string())
R = X.iloc[:N][keep[:N]]
Z = X.iloc[N:].loc[[N + i for i in top]]
q = {c: ((Z[c] > R[c].quantile(.98)) | (Z[c] < R[c].quantile(.02))).mean() * 100 for c in X.columns}
print("\nstatistics where the most-fake series fall outside the real 2-98% range (% of them; 4% expected by chance)")
print(pd.Series(q).sort_values(ascending=False).head(8).round(1).to_string())
fig, ax = plt.subplots(4, 4, figsize=(18, 11))
show = [("FAKE", fake[i], info.loc[i]) for i in top[:12]] + [("REAL", real[j], None) for j in range(4)]
for a, (tag, (v, n0), d) in zip(ax.ravel(), show):
    a.plot(v, lw=.5, color="0.4")
    a.plot(pd.Series(v).rolling(30, center=True).median(), color="tab:blue", lw=1.2)
    a.axvline(n0, color="r", ls="--", lw=.8)
    t = tag if d is None else (f"FAKE {d.structure} lvl {d.level:.2f} sw {int(d.noise_switch)} "
                               f"L{int(bool(d.get('on_level')))}S{int(bool(d.get('on_spread')))}D{int(bool(d.get('on_dep')))}")
    a.set_title(t, fontsize=8)
    a.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig("outputs/probe/c2st_fakest.png", dpi=70)
