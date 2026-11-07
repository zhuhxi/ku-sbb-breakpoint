"""Event study (Box-Tiao / superposed epoch): align every series at its boundary, fit AR(10) on the pre part, and average the squared
standardised one-step prediction error (pre model kept fixed) for steps -30..+60 around the boundary. Real vs fake, label 1 / 0.
Squared errors clipped at 50 (heavy tails). Writes outputs/event_profile.{csv,png}."""
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, "probe"); import gen_eval2 as G, ku_gen as K
p, A, B = 10, 30, 60
def prof(x, n0):
    b, e = G.arfit(x[:n0]); sd = e.std(); full = x[n0 - A - p:n0 + B]
    if len(full) < A + B + p: return None
    X = np.column_stack([np.ones(len(full) - p)] + [full[p - k:len(full) - k] for k in range(1, p + 1)])
    return np.minimum(((full[p:] - X @ b) / sd) ** 2, 50)
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"].astype(int)
S = pd.read_parquet("data/train_series.parquet"); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"])
out = Parallel(n_jobs=32, batch_size=64)(delayed(lambda k, v, n0: (L[k], prof(v, n0)))(k, g["value"].to_numpy(float), int((g.period == 0).sum())) for k, g in S.groupby("id", sort=False))
P = G.parse("outputs/factory/ku_gen_size_three-bin.json")
def fk(s, y):
    x, n0, h = K.gen(P, 7_000_000 + 2 * s + y, bool(y)); return y, prof(x, n0), h["kind"], h["family"], h["lvl_mode"]
fo = Parallel(n_jobs=32, batch_size=64)(delayed(fk)(s, y) for s in range(12000) for y in (0, 1))
t = np.arange(-A, B); rows = {}
for y in (0, 1):
    rows[f"real_{y}"] = np.mean([v for l, v in out if l == y and v is not None], axis=0)
    rows[f"fake_{y}"] = np.mean([v for l, v, *_ in fo if l == y and v is not None], axis=0)
for kd in ("level", "spread", "dep", "shape"):
    rows[f"fake_1_{kd}"] = np.mean([v for l, v, k, *_ in fo if l == 1 and k == kd and v is not None], axis=0)
D = pd.DataFrame(rows, index=t); D.index.name = "step"; D.to_csv("outputs/event_profile.csv")
for c in ("real_1", "real_0", "fake_1", "fake_0"):
    v = D[c]; print(f"{c:8s} pre(-30..-1) {v[v.index < 0].mean():.3f} | step0 {v[0]:.3f} | 1-5 {v[1:6].mean():.3f} | 6-20 {v[6:21].mean():.3f} | 21-60 {v[21:].mean():.3f}")
print("excess label1-label0  real: step0 %.3f 1-5 %.3f 6-20 %.3f 21-60 %.3f" % tuple((D.real_1 - D.real_0)[s].mean() for s in (slice(0,0), slice(1,5), slice(6,20), slice(21,59))))
print("excess label1-label0  fake: step0 %.3f 1-5 %.3f 6-20 %.3f 21-60 %.3f" % tuple((D.fake_1 - D.fake_0)[s].mean() for s in (slice(0,0), slice(1,5), slice(6,20), slice(21,59))))
for kd in ("level", "spread", "dep", "shape"):
    v = D[f"fake_1_{kd}"]; print(f"fake_1 {kd:6s} step0 {v[0]:.3f} | 1-5 {v[1:6].mean():.3f} | 6-20 {v[6:21].mean():.3f} | 21-60 {v[21:].mean():.3f}")
