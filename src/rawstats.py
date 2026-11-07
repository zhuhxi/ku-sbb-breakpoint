"""datatech-style raw-scale pool: 12 statistics x {RAW, CUMSUM, DIFF} x {left, right, whole}.

Unlike features.py, nothing here is normalised by the before segment, so whole-series level/scale
information (where first-edition "magic" lived) is available to the interaction search.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats


def _stats(x):
    if len(x) < 4:
        return {}
    q05, q25, q50, q75, q95 = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    m, s = x.mean(), x.std()
    return {"mean": m, "std": s, "cv": s / (abs(m) + 1e-9), "median": q50, "mad": np.median(np.abs(x - q50)),
            "min": x.min(), "max": x.max(), "iqr": q75 - q25, "q05": q05, "q95": q95,
            "skew": stats.skew(x), "kurt": stats.kurtosis(x)}


def _one(sid, v, p):
    n0 = int((p == 0).sum())
    f = {"id": sid}
    for tname, t, b in (("RAW", v, n0), ("CUMSUM", np.cumsum(v), n0), ("DIFF", np.diff(v), n0 - 1)):
        for seg, x in (("left", t[:b]), ("right", t[b:]), ("whole", t)):
            for k, val in _stats(x).items():
                f[f"{tname}_{k}_{seg}"] = val
    return f


def build_rawstats(df, n_jobs=-1):
    df = df.sort_values(["id", "time"], kind="stable")
    groups = [(sid, g["value"].to_numpy(float), g["period"].to_numpy(int)) for sid, g in df.groupby("id", sort=False)]
    return pd.DataFrame(Parallel(n_jobs=n_jobs, batch_size=256)(delayed(_one)(*g) for g in groups)).set_index("id")


if __name__ == "__main__":
    for split in ("train", "test"):
        R = build_rawstats(pd.read_parquet(f"data/{split}_series.parquet"))
        R.to_parquet(f"outputs/raw_{split}.parquet")
        print(split, R.shape)
