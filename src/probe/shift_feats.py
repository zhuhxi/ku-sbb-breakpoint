"""Split moved EARLIER by k (post part starts at n0-k): AR(5) residual log-sd ratio, mean shift (pre-sd units), acf1 change,
on raw z and AR residuals, k in 0,3,6,10 + differences vs k=0. -> outputs/factory/shift.parquet"""
import numpy as np, pandas as pd
rows = {}
def one(v, n0):
    d = {}
    x = v[:max(n0 - 10, 20)]; mu, sd = x.mean(), x.std() + 1e-9; z = (v - mu) / sd; q = 5
    Zl = np.column_stack([np.ones(len(z) - q)] + [z[q - j - 1:len(z) - j - 1] for j in range(q)])
    m = len(x) - q; c = np.linalg.lstsq(Zl[:m], z[q:len(x)], rcond=None)[0]; e = np.r_[np.zeros(q), z[q:] - Zl @ c]
    ac = lambda u: np.corrcoef(u[:-1], u[1:])[0, 1]
    for k in (0, 3, 6, 10):
        s = n0 - k; a, b = z[q:s], z[s:]; ea, eb = e[q:s], e[s:]
        d[f"sh_lsd{k}"] = np.log(eb.std() / (ea.std() + 1e-9) + 1e-9); d[f"sh_mu{k}"] = abs(b.mean() - a.mean()) / (a.std() + 1e-9)
        d[f"sh_ac{k}"] = ac(b) - ac(a); d[f"sh_lsd20_{k}"] = np.log(eb[:20].std() / (ea.std() + 1e-9) + 1e-9)
    for k in (3, 6, 10):
        for f in ("lsd", "mu", "ac", "lsd20_"): d[f"sh_d{f}{k}"] = d[f"sh_{f}{k}"] - d[f"sh_{f}0"]
    return d
for p in ("data/train_series.parquet", "data/test_series.parquet"):
    S = pd.read_parquet(p); S["id"] = S.id.astype(str); S = S.sort_values(["id", "time"], kind="stable")
    for i, g in S.groupby("id", sort=False): rows[i] = one(g.value.to_numpy(float), int((g.period.to_numpy() == 0).sum()))
D = pd.DataFrame.from_dict(rows, orient="index"); D.index.name = "id"; D.to_parquet("outputs/factory/shift.parquet"); print(D.shape)
