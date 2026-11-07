"""Where do the too-steep slopes of generated break series come from? Pre-boundary "trend slope" statistic
(probe/hidden_diag.py) per bin: real break series vs menu "L abc40 ns35", and generated split by what was added
(whole-series trend layer, drift from the boundary, neither).

    python probe/slope_why.py
"""
import sys

sys.argv = [sys.argv[0], "--menu", "L abc40 ns35", "--only-break", "1"]
import importlib.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

s = importlib.util.spec_from_file_location("bc", "probe/break_c2st.py")
bc = importlib.util.module_from_spec(s); s.loader.exec_module(bc)
N = 3000
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
for y in (1, 0):
    ids = set(np.random.default_rng(11).choice(lab.index[lab == y], N, False))
    real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
    out = Parallel(n_jobs=30)(delayed(bc.cb.gen)(bc.MENU, 6 * 10 ** 6 + i, bool(y), "info") for i in range(N))
    R = pd.DataFrame(Parallel(n_jobs=30)(delayed(bc.feats)(v, n0) for v, n0 in real))
    G = pd.DataFrame(Parallel(n_jobs=30)(delayed(bc.feats)(x, n0) for x, n0, _ in out))
    I = pd.DataFrame([d for _, _, d in out])
    R, keep = R[R.cls.isin(["WN", "AR1", "AR2"])], G.cls.isin(["WN", "AR1", "AR2"]).to_numpy()
    G, I = G[keep], I[keep]
    bins = [0, .25, .5, 1, 2, 4, 99]
    cut = lambda D: pd.cut(D["trend slope"].abs(), bins, right=False).value_counts(normalize=True).sort_index() * 100
    T = pd.DataFrame({"real": cut(R), "generated": cut(G), "gen: trend layer": cut(G[I.trend.to_numpy()]),
                      "gen: drift": cut(G[I.drift.to_numpy()]), "gen: neither": cut(G[~(I.trend | I.drift).to_numpy()])})
    print(f"\n{'break' if y else 'no-break'} series: |trend slope| per bin (% of series); trend layer on {I.trend.mean() * 100:.0f}%, drift on {I.drift.mean() * 100:.0f}%")
    print(T.round(1).to_string())
