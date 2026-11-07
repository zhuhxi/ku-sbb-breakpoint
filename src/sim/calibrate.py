"""Compare simulated and real before/after changes; used to tune the break knobs.

    python sim/calibrate.py '{"p": [0.3, 0.5, 0.1, 0.1], "s_mean": 0.5, "s_var": 0.3, "s_ar": 0.2}'

Prints, for real and simulated data: the no-break 99% thresholds of each change statistic and
the share of breaks exceeding the REAL thresholds (the "break type" fingerprint from gen1.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simlib import fit_ar, make_donors, simulate  # noqa: E402


def seg(x):
    b, e = fit_ar(x)
    return dict(mean=x.mean(), std=x.std(), phi_sum=b[1:].sum(), rstd=e.std(), rkurt=stats.kurtosis(e), rskew=stats.skew(e))


def changes(df):
    rows = {}
    for sid, g in df.groupby("id", sort=False):
        v, p = g["value"].to_numpy(), g["period"].to_numpy()
        a, b = seg(v[p == 0]), seg(v[p == 1])
        rows[sid] = {"mean": (b["mean"] - a["mean"]) / np.sqrt((a["std"] ** 2 + b["std"] ** 2) / 2),
                     "lstd": np.log(b["rstd"] / a["rstd"]), "phi": b["phi_sum"] - a["phi_sum"],
                     "kurt": b["rkurt"] - a["rkurt"], "skew": b["rskew"] - a["rskew"], "phi0": a["phi_sum"]}
    return pd.DataFrame(rows).T


def real_donors():
    lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
    d = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
    d = d[d["id"].isin(lab.index[lab == 0])]
    return [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in d.groupby("id", sort=False)]


def fingerprint(C, y, thr):
    st = C.pop("phi0") < 0.9 if "phi0" in C else True   # stationary series only (random walks swamp the mean)
    N, B = C[(y == 0) & st], C[(y == 1) & st]
    E = B.abs() > thr
    return {"thr99_nobreak": N.abs().quantile(0.99).round(3).to_dict(),
            "exceed": {**E.mean().round(3).to_dict(), "any": round(E.any(axis=1).mean(), 3)}}


if __name__ == "__main__":
    knobs = json.loads(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    S = pd.read_parquet("outputs/gen_segstats.parquet")
    S = S[S.resc == 0]
    Creal = pd.DataFrame({"mean": (S.mean_1 - S.mean_0) / np.sqrt((S.std_0 ** 2 + S.std_1 ** 2) / 2),
                          "lstd": np.log(S.rstd_1 / S.rstd_0), "phi": S.phi_sum_1 - S.phi_sum_0,
                          "kurt": S.rkurt_1 - S.rkurt_0, "skew": S.rskew_1 - S.rskew_0, "phi0": S.phi_sum_0})
    thr = Creal[(S.label == 0) & (Creal.phi0 < 0.9)].drop(columns="phi0").abs().quantile(0.99)
    print("real:", fingerprint(Creal.copy(), S.label, thr))
    donors = make_donors(real_donors())
    parts = Parallel(n_jobs=24)(delayed(simulate)(donors, n // 24, knobs, seed=s) for s in range(24))
    df = pd.concat([p[0] for p in parts]); y = pd.concat([p[1] for p in parts]); bt = pd.concat([p[2] for p in parts])
    C = changes(df).loc[y.index]
    print("sim: ", fingerprint(C.copy(), y, thr))
    C = C[C.pop("phi0") < 0.9]; bt = bt.loc[C.index]
    E = C.abs() > thr
    print("sim exceed by break type:\n", E.groupby(bt).mean().round(3).assign(any=E.any(axis=1).groupby(bt).mean().round(3)))
