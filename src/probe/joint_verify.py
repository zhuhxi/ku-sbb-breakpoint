"""Re-test rows of outputs/probe/joint_search.csv with fresh generator seeds and fresh real samples (the search
picked the best of many noisy scores, so the winner's score is optimistic).

    python probe/joint_verify.py ROW [ROW ...]
"""
import sys

rows = [int(a) for a in sys.argv[1:]]
sys.argv = [sys.argv[0], "--k", "0"]
import importlib.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

s = importlib.util.spec_from_file_location("js", "probe/joint_search.py")
js = importlib.util.module_from_spec(s)
src = open("probe/joint_search.py").read().split("lab = pd.read_csv")[0]
exec(src, js.__dict__)
bc, menu = js.bc, js.menu
C = pd.read_csv(js.OUT)
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet")
for rep in range(3):
    ids = set(np.random.default_rng(100 + rep).choice(lab.index[lab == 1], 1500, False))
    real = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in S[S.id.isin(ids)].sort_values(["id", "time"]).groupby("id")]
    out = []
    for row in rows:
        c = C.iloc[row][list(js.RANGES)].to_dict()
        fake = Parallel(n_jobs=30)(delayed(bc.cb.gen)(menu(c), 7 * 10 ** 6 + 10 ** 4 * rep + i, True, True) for i in range(1500))
        out.append(f"row {row}: {bc.c2st(real, fake)[0]:.3f}")
    print(f"repeat {rep}: " + "   ".join(out), flush=True)
