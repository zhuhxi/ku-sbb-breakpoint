"""Do label and generator metadata depend on each other? (1) boundary position rule by label, (2) label rate by
fitted structure / noise family / length.

    python probe/meta_label.py
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import roc_auc_score

t0 = time.time()
SP = "/tmp/claude-1006/-mnt-disk2-zhuhx-yolov10/52e90985-5ae5-4402-bf67-8e7f98c2f063/scratchpad/"
D = pd.read_parquet(SP + "nn0_train.parquet").join(pd.read_csv("data/train_labels.csv").set_index("id"))
D = D.join(pd.read_parquet("outputs/factory/genA.parquet")[["g_struct", "g_noise"]])
D["frac"] = D.n0 / D.n
y = D.label
print("1) boundary position by label")
for c in ["n", "n0", "frac"]:
    print(f"   {c:5s} AUC {roc_auc_score(y, D[c]):.3f}  KS p {ks_2samp(D.loc[y == 0, c], D.loc[y == 1, c]).pvalue:.3f}")
for lab in (0, 1):
    f = D.loc[y == lab, "frac"]
    print(f"   label {lab}: frac min {f.min():.4f} max {f.max():.4f}  q01 {f.quantile(.01):.4f} q99 {f.quantile(.99):.4f}")
# rounding rule of n0: floor(u*n) vs round(u*n) leaves different edge behaviour; check n0 vs int(.3n), int(.7n)
for lab in (0, 1):
    E = D[y == lab]
    print(f"   label {lab}: n0 == floor(.3n) {np.mean(E.n0 == np.floor(.3 * E.n)):.4f}  n0 < ceil(.3n) {np.mean(E.n0 < np.ceil(.3 * E.n)):.4f}"
          f"  n0 > floor(.7n) {np.mean(E.n0 > np.floor(.7 * E.n)):.4f}  n0 even {np.mean(E.n0 % 2 == 0):.3f}")
print("   frac histogram 20 bins, label0:", np.histogram(D.loc[y == 0, "frac"], 20, (.3, .7))[0])
print("   frac histogram 20 bins, label1:", np.histogram(D.loc[y == 1, "frac"], 20, (.3, .7))[0])
print("2) label rate by group")
ST = ["WN", "AR1", "AR2", "MA1", "ARMA11", "RW", "SIN+WN", "SIN+AR1"]
NZ = ["normal", "t", "bimodal"]
for c, names in [("g_struct", ST), ("g_noise", NZ)]:
    t = pd.crosstab(D[c], y)
    print(f"   {c}: chi2 p {chi2_contingency(t)[1]:.3g}  ", {names[int(k)]: f"{r[1] / r.sum():.3f} (n={r.sum()})" for k, r in t.iterrows()})
nb = pd.cut(D.n, [255, 400, 550, 700, 850, 1024])
t = pd.crosstab(nb, y)
print(f"   length bins: chi2 p {chi2_contingency(t)[1]:.3g}  ", {str(k): round(r[1] / r.sum(), 3) for k, r in t.iterrows()})
print(f"{time.time() - t0:.1f}s")
