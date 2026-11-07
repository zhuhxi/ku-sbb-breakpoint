"""Plug-in generator likelihood ratio (probe/oracle_plugin.py estimate + oracle_ceiling.score) on every REAL series,
total and per kind (spread / level / dep / shape). WN / AR1 model for every series (no family switch).
    python probe/pluglr_real.py  -> outputs/factory/pluglr.parquet"""
import sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, "probe")
import oracle_ceiling as O, oracle_plugin as OP


def one(x, n0, s):
    try:
        return O.score(x, n0, OP.estimate(x, n0), s, parts=True)
    except Exception:
        return {}


if __name__ == "__main__":
    import pluglr_real as PR
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]); S["id"] = S["id"].astype(str)
    S = S.sort_values(["id", "time"])
    items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    R = Parallel(n_jobs=30, batch_size=16)(delayed(PR.one)(v, n0, i) for i, (_, v, n0) in enumerate(items))
    F = pd.DataFrame(R, index=[k for k, _, _ in items]).astype(float)
    F.to_parquet("outputs/factory/pluglr.parquet"); print(F.shape, F.isna().mean().round(3).to_dict(), f"{time.time() - t0:.0f}s")
