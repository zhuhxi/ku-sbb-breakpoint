"""5-10 s check of the claim 'ADIA-style raw before/after stats fail on KU because KU series have structure; removing the
structure first (AR(5) residuals fitted on the pre part) is what makes the comparison work'. 8000 train series."""
import time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ks_2samp, levene
from sklearn.metrics import roc_auc_score
t0 = time.time()
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"].iloc[:8000]
S = pd.read_parquet("data/train_series.parquet"); S["id"] = S["id"].astype(str)
S = S[S.id.isin(set(L.index))].sort_values(["id", "time"])
def stats(a, b):
    return [abs(b.mean() - a.mean()) / a.std(), abs(np.log(b.std() / a.std())), ks_2samp(a, b).statistic, levene(np.abs(a), np.abs(b))[0]]
def one(v, n0):
    X = np.column_stack([np.ones(len(v) - 5)] + [v[5 - k:len(v) - k] for k in range(1, 6)])
    c = np.linalg.lstsq(X[:n0 - 5], v[5:n0], rcond=None)[0]; r = v[5:] - X @ c
    return [np.corrcoef(v[1:n0], v[:n0 - 1])[0, 1]] + stats(v[:n0], v[n0:]) + stats(r[:n0 - 5], r[n0 - 5:])
G = [(g["value"].to_numpy(float), int((g["period"] == 0).sum())) for _, g in S.groupby("id", sort=False)]
R = np.array(Parallel(n_jobs=30, batch_size=64)(delayed(one)(v, n0) for v, n0 in G)); y = L.loc[S.id.unique()].to_numpy()
print(f"KU lag-1 ACF of the pre part: median |acf| {np.median(np.abs(R[:, 0])):.2f}, share |acf| > .3: {np.mean(np.abs(R[:, 0]) > .3):.2f}  (ADIA archive: ~0)")
nm = ["mean shift", "log sd ratio", "KS", "Levene |x|"]
for grp, m in (("all", np.ones(len(y), bool)), ("near-iid |acf|<.1", np.abs(R[:, 0]) < .1), ("structured |acf|>.5", np.abs(R[:, 0]) > .5)):
    print(f"{grp:20s} n={m.sum():5d} | " + "  ".join(f"{nm[i]} raw {roc_auc_score(y[m], R[m, 1 + i]):.3f} -> resid {roc_auc_score(y[m], R[m, 5 + i]):.3f}" for i in range(4)))
print(f"{time.time() - t0:.1f}s")
