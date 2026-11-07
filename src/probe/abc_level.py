"""ABC for the level-jump size of real break series. Simulate menu "L tuned" with a wide prior on the level jump
(on with prob .5, size U(0, 3) noise sd), match each real break series to its 50 nearest simulations on the
before-vs-after statistics (probe/hidden_diag.py), then deconvolve: re-weight the prior over size bins until it
equals the pooled posterior (EM over the fixed neighbour sets).

    python probe/abc_level.py
"""
import sys
import time

sys.argv = [sys.argv[0], "--menu", "L tuned", "--only-break", "1"]
import importlib.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

s = importlib.util.spec_from_file_location("bc", "probe/break_c2st.py")
bc = importlib.util.module_from_spec(s); s.loader.exec_module(bc)
NSIM, NREAL, K = 20000, 3000, 50
STATS = ["level", "robust level (median)", "local: first 30 after vs last 30 before", "spread", "robust spread (MAD)",
         "noise shape (kurtosis)", "dep", "n", "frac"]


def sim(seed):
    x, n0, lv = bc.cb.gen(B, seed, True, "truth")
    f = bc.feats(x, n0)
    f["true"] = lv
    return f


B = dict(bc.cb.BREAKS["L tuned"], switches=dict(level=.5, spread=.35, dep=.2), level=(0.0, 3.0))
if __name__ == "__main__":
    t0 = time.time()
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    S = pd.read_parquet("data/train_series.parquet")
    ids = set(np.random.default_rng(11).choice(lab.index[lab == 1], NREAL, False))
    real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
    R = pd.DataFrame(Parallel(n_jobs=30)(delayed(bc.feats)(v, n0) for v, n0 in real))
    G = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(sim)(7 * 10 ** 6 + i) for i in range(NSIM)))
    print(f"simulated {NSIM} ({time.time() - t0:.0f}s)")
    R, G = R[R.cls.isin(["WN", "AR1", "AR2"])], G[G.cls.isin(["WN", "AR1", "AR2"])]
    prep = lambda D: np.column_stack([D[c].abs() if "level" in c or "local" in c else D[c] for c in STATS]).astype(float)
    XG, XR = prep(G), prep(R)
    mu, sc = np.median(XG, 0), np.median(np.abs(XG - np.median(XG, 0)), 0) + 1e-9
    nn = NearestNeighbors(n_neighbors=K).fit((XG - mu) / sc)
    idx = nn.kneighbors((XR - mu) / sc, return_distance=False)
    # size bins: 0 = no level change, then 0.1-wide bins up to 3
    edges = np.r_[0, np.arange(0.1, 3.01, 0.1)]
    tb = np.where(G.true.to_numpy() == 0, 0, 1 + np.minimum(np.digitize(G.true.to_numpy(), edges[1:-1]), 29))
    nb = 31
    prior0 = np.bincount(tb, minlength=nb) / len(tb)
    C = np.stack([np.bincount(tb[i], minlength=nb) for i in idx]).astype(float)   # neighbour counts per real series
    w = prior0.copy()
    for _ in range(500):                                       # EM deconvolution of the population distribution
        P = C * (w / prior0)
        P /= P.sum(1, keepdims=True)
        w = P.mean(0)
    print(f"real series used: {len(XR)}   P(no level change) {w[0]:.2f}")
    on = w[1:] / w[1:].sum()
    print("size bin (noise sd)   share of level changes")
    for j in range(30):
        print(f"  {edges[j]:.1f}-{edges[j + 1]:.1f}   {on[j] * 100:5.1f}%  {'#' * int(on[j] * 200)}")
    print(f"total {time.time() - t0:.0f}s")
