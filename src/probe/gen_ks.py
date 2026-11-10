"""KS ranking of hand statistics, real vs fake (per label).   python probe/gen_ks.py PF.json ['dict(updates)'] [TOP]"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import ks_2samp
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 12; sys.argv = sys.argv[:3]; os.environ.setdefault("SEED", "99000000")
exec(open("probe/gen_c2st.py").read().split("cols = [c for c in R")[0])
rows = []
for y in (0, 1):
    for c in [c for c in R if c != "y"]:
        a, b_ = R[R.y == y][c].replace([np.inf, -np.inf], np.nan).dropna(), F[F.y == y][c].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append((y, c, ks_2samp(a, b_).statistic, np.round(np.quantile(a, [.05, .5, .95]), 3), np.round(np.quantile(b_, [.05, .5, .95]), 3)))
D = pd.DataFrame(rows, columns=["y", "stat", "KS", "real q05/50/95", "fake q05/50/95"]).sort_values("KS", ascending=False)
pd.set_option("display.width", 220); print(D.head(TOP).to_string(index=False))
