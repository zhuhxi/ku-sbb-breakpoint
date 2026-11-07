"""List every training series the current best (LGB + inter arms 1, 2, OOF rank blend) gets wrong.
Labels are 50/50, so the cut is the median score: label 1 below it = missed break, label 0 above it = false alarm.
"Confident" errors use the 25% / 75% score quantiles. Writes outputs/error_cases.csv (id, label, score, kind,
confident, structure, n, n0).

    python probe/error_cases.py
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

STRUCT = ["WN", "AR1", "AR2", "MA1", "ARMA11", "RW", "SIN+WN", "SIN+AR1"]
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = L.to_numpy()
r = lambda v: rankdata(v) / len(v)
I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
s = pd.Series((r(np.load("outputs/factory/confirm_combo10gnj.npz")["oof"]) + r(I[1]) + r(I[2])) / 3, index=L.index)
A = pd.read_parquet("outputs/factory/genA.parquet")
A.index = A.index.astype(str)
S = pd.read_parquet("data/train_series.parquet", columns=["id", "period"])
S["id"] = S["id"].astype(str)
nn = S.groupby("id")["period"].agg(n="size", n0=lambda p: int((p == 0).sum()))
med, q25, q75 = s.median(), s.quantile(.25), s.quantile(.75)
D = pd.DataFrame({"label": y, "score": s.round(4)}, index=L.index)
D["kind"] = np.where((y == 1) & (s < med), "missed break", np.where((y == 0) & (s >= med), "false alarm", "correct"))
D["confident"] = ((y == 1) & (s < q25)) | ((y == 0) & (s > q75))
D["structure"] = A["g_struct"].reindex(L.index).map(dict(enumerate(STRUCT)))
D = D.join(nn)
E = D[D.kind != "correct"].sort_values("score")
E.to_csv("outputs/error_cases.csv")
print(f"all {len(D)}: correct {(D.kind == 'correct').sum()}, missed breaks {(D.kind == 'missed break').sum()}, false alarms {(D.kind == 'false alarm').sum()}  "
      f"(accuracy at the median cut {(D.kind == 'correct').mean():.3f})")
print(f"confident errors: missed (score < q25) {((D.kind == 'missed break') & D.confident).sum()}, false alarms (score > q75) {((D.kind == 'false alarm') & D.confident).sum()}")
T = pd.crosstab(D.structure, D.kind)
T["error rate"] = ((T["missed break"] + T["false alarm"]) / T.sum(1)).round(3)
print(T.to_string())
