"""Which kind of series has the wrong size? Real label 0 vs generated no-break, binned by lag-1 autocorrelation
(proxy for structure: ~0 white noise, middle AR, ~1 random walk / local level); per bin share, median log sd, and the
generator's structures in that bin.

    python probe/scale_by_acf.py [N] [MENU]
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
MENU = sys.argv[2] if len(sys.argv) > 2 else "ok14"
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("cb", "probe/calib_break.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


def f(x):
    z = x - x.mean()
    return np.log(x.std()), z[1:] @ z[:-1] / (z @ z)


def g(s):
    x, n0, d = cb.gen(cb.BREAKS[MENU], 10_000 + s, False, "info")
    return (*f(x), d["structure"])


L = pd.read_csv("data/train_labels.csv", dtype={"id": str})
S = pd.read_parquet("data/train_series.parquet")
S = S[S["id"].astype(str).isin(set(L.loc[L.label == 0, "id"].to_numpy()[:N]))].sort_values(["id", "time"])
R = pd.DataFrame([f(g_["value"].to_numpy(np.float64)) for _, g_ in S.groupby("id", sort=False)], columns=["lsd", "acf"])
F = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(g)(s) for s in range(N)), columns=["lsd", "acf", "st"])
bins = [-1, -.2, .2, .6, .9, .98, 1]
R["b"], F["b"] = pd.cut(R.acf, bins), pd.cut(F.acf, bins)
for b in R.b.cat.categories:
    r, f_ = R[R.b == b], F[F.b == b]
    st = f_.st.value_counts(normalize=True).head(3).round(2).to_dict()
    print(f"acf1 {str(b):14s} share real {len(r) / len(R):.3f} fake {len(f_) / len(F):.3f} | median log sd real {r.lsd.median():+.3f} "
          f"fake {f_.lsd.median():+.3f} (diff {f_.lsd.median() - r.lsd.median():+.3f}) | fake structures {st}")
