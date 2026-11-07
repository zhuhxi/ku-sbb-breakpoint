"""Which simulator change brings no-break features closest to real? Cheap comparison on ~2800 series."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import simlib
from features import build_features
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
d = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
ids0 = lab.index[lab == 0][:4000]
G = [(g["value"].to_numpy(), int((g["period"] == 0).sum())) for _, g in d[d.id.isin(set(ids0))].groupby("id", sort=False)]
real = pd.read_parquet("outputs/feat_train.parquet").loc[ids0]
C = ["ad", "cusum_max", "nl_t_p", "acf_port", "pe5_b", "nl_lev_p", "lr_std"]
print("real no-break  ", real[C].median().round(3).to_dict())

def run(name, donors, block=16):
    simlib.BLOCK = block
    df, y, _ = simlib.simulate(donors, 2800, {**simlib.KNOBS, "p": [0.25, 0.25, 0.25, 0.25]}, seed=5)
    F = build_features(df); F = F[y.loc[F.index] == 0]
    print(f"{name:15s}", F[C].median().round(3).to_dict(), flush=True)

base = simlib.make_donors(G)
run("current", base)
run("iid boot", base, block=1)
shr = [{**dn, "phi": dn["phi"] * (np.abs(dn["phi"].sum()) > 0.15)} for dn in base]
run("phi=0 if weak", shr)
iid = []
for x, n0 in G:
    iid.append({"c": x.mean(), "d": 0.0, "phi": np.zeros(simlib.P), "e": np.random.default_rng(0).normal(0, x.std(), 600), "unit": False,
                "x0": np.full(simlib.P + 1, x.mean()), "sd": x.std(), "n": len(x), "n0": n0})
run("iid normal", iid, block=1)
