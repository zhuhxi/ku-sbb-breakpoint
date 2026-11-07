"""KU real vs our generator: signs of 'pasted segments' vs 'one process with a parameter switch'. Per series:
jump  = |x[n0]-x[n0-1]| / median |dx| (pre)      -> pasted pieces give a big boundary step (esp. persistent families)
start = var of first 20 post diffs / var of later post diffs   -> fresh segment without warm-up shows a start transient
phi   = lag-1 autocorrelation pre vs post (corr across series) -> pasted: post params independent of pre
multi = number of properties changed (|z|>3: mean, log sd, acf1, kurt of diffs) -> pasted: several at once"""
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, "probe"); import gen_eval2 as G, ku_gen as K
from scipy.stats import kurtosis
ac = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]
def f(x, n0):
    d = np.diff(x); dp, dq = d[:n0 - 1], d[n0:]; m = len(dq)
    jump = abs(x[n0] - x[n0 - 1]) / (np.median(np.abs(dp)) + 1e-12)
    start = dq[:20].var() / (dq[20:].var() + 1e-12)
    a0, a1 = ac(x[:n0]), ac(x[n0:])
    se = lambda n: 1 / np.sqrt(n)
    z = [abs(dq.mean() - dp.mean()) / np.sqrt(dp.var() / len(dp) + dq.var() / m),
         abs(np.log(dq.std() / dp.std())) / np.sqrt(.5 / len(dp) + .5 / m),
         abs(ac(dq) - ac(dp)) / np.sqrt(se(len(dp)) ** 2 + se(m) ** 2),
         abs(kurtosis(dq) - kurtosis(dp)) / np.sqrt(24 / len(dp) + 24 / m)]
    return dict(jump=jump, start=start, a0=a0, a1=a1, multi=int(sum(v > 3 for v in z)))
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"].astype(int)
ids = set(L.sample(8000, random_state=3).index)
S = pd.read_parquet("data/train_series.parquet"); S["id"] = S["id"].astype(str); S = S[S.id.isin(ids)].sort_values(["id", "time"])
R = pd.DataFrame(Parallel(n_jobs=32, batch_size=64)(delayed(lambda k, v, n0: dict(f(v, n0), y=L[k]))(k, g["value"].to_numpy(float), int((g.period == 0).sum())) for k, g in S.groupby("id", sort=False)))
P = G.parse("outputs/factory/ku_gen_size_three-bin.json")
F = pd.DataFrame(Parallel(n_jobs=32, batch_size=64)(delayed(lambda s, y: dict(f(*K.gen(P, 6_000_000 + 2*s + y, bool(y))[:2]), y=y))(s, y) for s in range(4000) for y in (0, 1)))
for nm, D in (("real", R), ("fake", F)):
    print(f"=== {nm}")
    for y in (0, 1):
        E = D[D.y == y]
        print(f"  label {y}: jump median {E.jump.median():.2f} q99 {E.jump.quantile(.99):.1f} | start ratio median {E.start.median():.3f} "
              f"| corr(acf pre, acf post) {np.corrcoef(E.a0, E.a1)[0,1]:.3f} | #props changed: " +
              " ".join(f"{k}:{v:.3f}" for k, v in E.multi.value_counts(normalize=True).sort_index().items()))
