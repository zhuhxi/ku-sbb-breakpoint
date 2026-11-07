"""Do per-side STANDARDISED shape tests catch the missed breaks better? Missed (probe/miss_scan.py definition) vs
3000 negatives: single AUC of two-sample shape statistics on raw values vs on each side standardised by its own
mean/sd ("z") or median/MAD ("r"); plus L-moment (L-skew, L-kurt) and moment differences.

    python probe/shape_std.py
"""
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as ss
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = L.to_numpy()
r = lambda v: rankdata(v) / len(v)
I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
s = (r(np.load("outputs/factory/confirm_combo10gnj.npz")["oof"]) + r(I[1]) + r(I[2])) / 3
miss = L.index[(y == 1) & (s < np.median(s[y == 0]))]
neg = L.index[y == 0].to_series().sample(3000, random_state=0).index
ids = set(miss) | set(neg)
S = pd.read_parquet("data/train_series.parquet")
S["id"] = S["id"].astype(str)
S = S[S.id.isin(ids)].sort_values(["id", "time"])


def lmom(x):
    x = np.sort(x)
    n = len(x)
    i = np.arange(n)
    b0, b1 = x.mean(), (i / (n - 1) * x).mean()
    b2 = (i * (i - 1) / ((n - 1) * (n - 2)) * x).mean()
    b3 = (i * (i - 1) * (i - 2) / ((n - 1) * (n - 2) * (n - 3)) * x).mean()
    l2, l3, l4 = 2 * b1 - b0, 6 * b2 - 6 * b1 + b0, 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l3 / l2, l4 / l2


def tests(a, b, tag):
    return {f"{tag}_ks": ss.ks_2samp(a, b).statistic, f"{tag}_ad": ss.anderson_ksamp([a, b]).statistic,
            f"{tag}_energy": ss.energy_distance(a, b), f"{tag}_wass": ss.wasserstein_distance(a, b)}


def one(k, x, n0):
    a, b = x[:n0], x[n0:]
    za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    mad = lambda v: np.median(np.abs(v - np.median(v))) + 1e-12
    ra, rb = (a - np.median(a)) / mad(a), (b - np.median(b)) / mad(b)
    la, lb = lmom(a), lmom(b)
    return {"id": k, **tests(a, b, "raw"), **tests(za, zb, "z"), **tests(ra, rb, "r"),
            "d_lskew": abs(lb[0] - la[0]), "d_lkurt": abs(lb[1] - la[1]),
            "d_skew": abs(ss.skew(b) - ss.skew(a)), "d_kurt": abs(ss.kurtosis(b) - ss.kurtosis(a))}


R = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(one)(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum()))
                                                    for k, g in S.groupby("id", sort=False))).set_index("id")
t = R.index.isin(miss)
print(f"missed {t.sum()} vs neg {(~t).sum()}")
print(pd.Series({c: roc_auc_score(t, R[c].fillna(R[c].median())) for c in R}).sort_values(ascending=False).round(3).to_string())
