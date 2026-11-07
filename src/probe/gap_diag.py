"""Diagnose the three C2ST gaps of the best2 generator (probe/c2st_best2.py), real train vs 24k best2 fakes with hidden parameters.
A) near20 (AR(10) fit on pre, mean squared standardised one-step error over the first 20 post points) vs the same over the WHOLE
   post part (lvar): onset problem (only near20 low) or size problem (both low)?  Fakes split by kind / level mode.
B) spread: log sd ratio post/pre of the AR(10) pre-model one-step errors; quantiles, share of increases, share |.|>.3, label 1 vs 0.
C) cv of |AR(10) residuals| (whole-series fit; Gaussian = .756, heavier tails higher): real vs fakes by noise type x GARCH.
    python probe/gap_diag.py"""
import json, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kurtosis
sys.path.insert(0, "probe")
from ku_gen import DEFAULT, gen
t0 = time.time()
B = {k: (tuple(v) if isinstance(DEFAULT[k], tuple) else v) for k, v in json.load(open("outputs/factory/ku_gen_best2.json")).items()}
P = dict(DEFAULT, **B); p = 10


def arfit(s):
    X = np.column_stack([np.ones(len(s) - p)] + [s[p - k:len(s) - k] for k in range(1, p + 1)])
    b = np.linalg.lstsq(X, s[p:], rcond=None)[0]; return b, s[p:] - X @ b


def one(x, n0):
    b, e = arfit(x[:n0]); sd = e.std()
    full = np.r_[x[n0 - p:], ]
    X = np.column_stack([np.ones(len(full) - p)] + [full[p - k:len(full) - k] for k in range(1, p + 1)])
    o = (full[p:] - X @ b) / sd
    _, ew = arfit(x); a = np.abs(ew)
    return dict(near20=np.mean(o[:20] ** 2), lvar=np.log(np.mean(o ** 2)), lsd=np.log(o.std()), cv=a.std() / a.mean(), kurt=kurtosis(ew))


L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
S = pd.read_parquet("data/train_series.parquet"); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"])
items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
R = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(one)(x, n0) for _, x, n0 in items), index=[k for k, _, _ in items])
R["label"] = L.loc[R.index].to_numpy()


def fk(s):
    x, n0, h = gen(P, 9_500_000 + s, bool(s % 2)); return dict(one(x, n0), label=s % 2, **{k: h[k] for k in ("family", "kind", "lvl_mode", "ntype", "garch", "log_k")})


F = pd.DataFrame(Parallel(n_jobs=30, batch_size=64)(delayed(fk)(s) for s in range(24000)))
q = lambda s: " ".join(f"{v:.2f}" for v in s.quantile([.25, .5, .75, .9]))
print(f"{time.time() - t0:.0f}s\nA) near20 / whole-post lvar, quantiles .25 .5 .75 .9")
for y in (0, 1):
    print(f"  label {y}: near20 real {q(R[R.label == y].near20)} | fake {q(F[F.label == y].near20)}")
    print(f"           lvar   real {q(R[R.label == y].lvar)} | fake {q(F[F.label == y].lvar)}")
f1 = F[F.label == 1]
print("  fake label 1 median near20 / lvar by kind:", f1.groupby("kind")[["near20", "lvar"]].median().round(3).to_dict("index"))
print("  fake level breaks by mode (0 step, 1 intercept, 2 late/ramp):", f1[f1.kind == "level"].groupby("lvl_mode")[["near20", "lvar"]].median().round(3).to_dict("index"))
print("B) log sd ratio post/pre of the pre-model errors")
for y in (0, 1):
    r, f = R[R.label == y].lsd, F[F.label == y].lsd
    print(f"  label {y}: real q {q(r)} up-share {np.mean(r > 0):.3f} |.|>.3 {np.mean(np.abs(r) > .3):.3f} | fake q {q(f)} up-share {np.mean(f > 0):.3f} |.|>.3 {np.mean(np.abs(f) > .3):.3f}")
fs = f1[f1.kind == "spread"]
print(f"  fake spread breaks: true log k up-share {np.mean(fs.log_k > 0):.3f}, |log k| q {q(fs.log_k.abs())}")
print("C) cv of |AR(10) residual| (Gaussian .756) and residual kurtosis, label 0")
r0, f0 = R[R.label == 0], F[F.label == 0]
print(f"  real cv q {q(r0.cv)} | fake cv q {q(f0.cv)}")
print(f"  real kurt q {q(r0["kurt"])} | fake kurt q {q(f0["kurt"])}")
print("  fake median cv / kurt / share by noise type (0 gauss, 1 t, 2 bimodal) x GARCH:")
print(f0.groupby(["ntype", "garch"]).agg(cv=("cv", "median"), kurt=("kurt", "median"), share=("cv", lambda s: len(s) / len(f0))).round(3).to_string())
print(f"  real share cv < .78 (Gaussian-like) {np.mean(r0.cv < .78):.3f} | fake {np.mean(f0.cv < .78):.3f};  real cv > .9 {np.mean(r0.cv > .9):.3f} | fake {np.mean(f0.cv > .9):.3f}")
print(f"{time.time() - t0:.0f}s")
