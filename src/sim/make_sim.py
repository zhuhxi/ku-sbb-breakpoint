"""Simulate labelled series from the no-break donors of one CV training fold and build their features.

    python sim/make_sim.py --fold 0 --n 48000      # donors = label-0 rows of fold 0's TRAINING part
    python sim/make_sim.py --fold all --n 48000    # donors = all label-0 train rows (final model)

Folds are StratifiedKFold(5, shuffle, 42) over train_labels.csv, as everywhere else, so simulated
rows used to train fold k never derive from fold k's validation series.
Writes outputs/sim/feat_{fold}.parquet (features.py columns + label + btype).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features  # noqa: E402
from simlib import KNOBS, make_donors, simulate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fold", default="0")
ap.add_argument("--n", type=int, default=48000)
ap.add_argument("--knobs", default=json.dumps(KNOBS))
ap.add_argument("--tag", default="")
args = ap.parse_args()

lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
if args.fold == "all":
    rows = lab.index
else:
    tr, _ = list(StratifiedKFold(5, shuffle=True, random_state=42).split(lab, lab))[int(args.fold)]
    rows = lab.index[tr]
don_ids = set(rows[lab.loc[rows].to_numpy() == 0])
d = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
d = d[d["id"].isin(don_ids)]
donors = make_donors([(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in d.groupby("id", sort=False)])
t = time.time()
knobs = json.loads(args.knobs)
seed0 = 1000 * (5 if args.fold == "all" else int(args.fold))
parts = Parallel(n_jobs=28)(delayed(simulate)(donors, args.n // 28, knobs, seed=seed0 + s, prefix=f"f{args.fold}_") for s in range(28))
df = pd.concat([p[0] for p in parts], ignore_index=True)
y = pd.concat([p[1] for p in parts]); bt = pd.concat([p[2] for p in parts])
print(f"fold {args.fold}: {len(donors)} donors -> {len(y)} series in {time.time() - t:.0f}s", flush=True)
t = time.time()
F = build_features(df)
F["label"], F["btype"] = y.loc[F.index], bt.loc[F.index]
Path("outputs/sim").mkdir(exist_ok=True)
F.to_parquet(f"outputs/sim/feat{args.tag}_{args.fold}.parquet")
print(f"features {F.shape} in {time.time() - t:.0f}s", flush=True)
