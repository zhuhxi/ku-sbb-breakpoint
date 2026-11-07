"""Which 2nd-place columns are new to us? max |Spearman| of each p2 column with the 888-column base (train rows).
Writes outputs/p2/novel.csv and per-family parquet files of the novel columns (max |rho| < .9)."""
import json, re, time
import numpy as np, pandas as pd
t = time.time()
exec(open("probe/param_quick.py").read().split("tr, va = next")[0].replace("name, vals = sys.argv[1], [float(v) for v in sys.argv[2:]]", ""))
P2 = pd.read_parquet("outputs/p2/xs.parquet").reindex(lab.index).replace([np.inf, -np.inf], np.nan)
def rz(D):
    R = D.rank().fillna(D.rank().mean()).to_numpy(np.float64); R -= R.mean(0); R /= R.std(0) + 1e-12; return R
A, B = rz(P2), rz(X)
C = np.abs(A.T @ B) / len(A)
mx = pd.Series(np.nanmax(C, 1), index=P2.columns)
def fam(c):
    m = re.match(r"w_(\w+?)_(\d)_(\d+)_", c)
    if m: return f"win_T{m.group(2)}"
    if c.startswith("g_"): return "global_T" + c[-1]
    return "tests"
F = pd.DataFrame({"maxrho": mx, "fam": [fam(c) for c in mx.index]})
F.to_csv("outputs/p2/novel.csv")
print(f"{X.shape[1]} base cols, {P2.shape[1]} p2 cols, {time.time() - t:.0f}s")
print("share with max|rho| < .9 / < .8:", round((mx < .9).mean(), 3), round((mx < .8).mean(), 3))
print(F.groupby("fam").maxrho.agg(n="size", median="median", novel09=lambda s: (s < .9).sum()))
for f, g in F[F.maxrho < .9].groupby("fam"):
    P2[g.index].to_parquet(f"outputs/p2/novel_{f}.parquet")
P2[F.index[F.maxrho < .9]].to_parquet("outputs/p2/novel_all.parquet")
