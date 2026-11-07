"""Write schema-matched fake KU-SBB files to data_fake/ for smoke-testing the pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
out = Path("data_fake")
out.mkdir(exist_ok=True)


def series(break_):
    n0, n1 = rng.integers(150, 400, 2)
    phi = rng.uniform(-0.5, 0.9)
    x = np.zeros(n0 + n1)
    eps = rng.standard_normal(n0 + n1)
    for t in range(1, len(x)):
        x[t] = phi * x[t - 1] + eps[t]
    if rng.random() < 0.3:  # temporary spike, not a break
        s = rng.integers(n0, n0 + n1 - 10)
        x[s:s + 5] += rng.normal(0, 4)
    if break_:
        kind = rng.integers(3)
        if kind == 0:
            x[n0:] += rng.normal(0, 0.6)
        elif kind == 1:
            x[n0:] *= rng.uniform(1.2, 1.8)
        else:
            x[n0:] = np.convolve(x[n0:], [0.5, 0.5], "same")
    return x, n0


def write(ids, path, labels=None):
    rows = []
    for i, sid in enumerate(ids):
        x, n0 = series(labels[i] if labels is not None else rng.integers(2))
        rows.append(pd.DataFrame({"id": sid, "time": np.arange(len(x)), "value": x,
                                  "period": (np.arange(len(x)) >= n0).astype(int)}))
    pd.concat(rows).to_parquet(path, index=False)


tr_ids = np.arange(1500)
te_ids = np.arange(100000, 100300)
y = rng.integers(0, 2, len(tr_ids))
write(tr_ids, out / "train_series.parquet", y)
write(te_ids, out / "test_series.parquet")
pd.DataFrame({"id": tr_ids, "label": y}).to_csv(out / "train_labels.csv", index=False)
pd.DataFrame({"id": te_ids, "break_probability": 0.5}).to_csv(out / "sample_submission.csv", index=False)
print("wrote", out)
