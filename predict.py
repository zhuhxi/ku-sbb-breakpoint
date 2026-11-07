"""KU-SBB team BreakPoint - CPU inference for the designated submission "blendD" (Kaggle public 0.86077).

blendD = rank( 2 * rank(mean of 11 pair-arm ranks) + 1 * rank(interaction arm "top 800") ), all ranks over the 4000 test rows.

Inputs (produced by the feature pipeline, see TRAINING.md):
  FEAT_DIR/feat_{train,test}.parquet, magic_b_*, harvest_*, harvest_keep.json, raw_{train,test}.parquet,
  FEAT_DIR/factory/{st_RES10,st_ARES10,fe,fe2,ar2,cost,genA,genB,gnoise,gjlr,semiauto_bag6w,semiauto_full,semiauto_bag6}.parquet,
  FEAT_DIR/factory/combo10gnjx_{train,test}.parquet
  DATA_DIR/train_labels.csv, DATA_DIR/sample_submission.csv (ids and order only)
Checkpoints: CKPT_DIR/<arm>/fold<k>/... (LightGBM text models + selected columns).

    python predict.py --data data --feat outputs --ckpt ckpt --out submission.csv [--threads 2]
"""
import argparse, gzip, json, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata

FAMS = ["st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost", "genA", "genB", "gnoise", "gjlr"]
PAT = r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_|^saf_"
# (tag, K, semi-automatic feature family) of the 11 pair arms in blendD
PAIR_ARMS = [("pair40", 40, "semiauto_bag6w"), ("pair60", 60, "semiauto_bag6w"), ("pair60saf", 60, "semiauto_full"),
             ("pair80", 80, "semiauto_bag6w"), ("pairb_50_1", 50, "semiauto_bag6w"), ("pairb_70_2", 70, "semiauto_bag6w"),
             ("pairb_60_3", 60, "semiauto_bag6"), ("pairb_55_5", 55, "semiauto_bag6w"), ("pairb_65_6", 65, "semiauto_bag6w"),
             ("pairb_60_7", 60, "semiauto_full"), ("pairb_75_8", 75, "semiauto_bag6w")]
OPS = ("mul", "add", "sub", "div", "rdiv", "mulb2", "mula2")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--feat", default="outputs"); ap.add_argument("--ckpt", default="ckpt")
ap.add_argument("--out", default="submission.csv"); ap.add_argument("--threads", type=int, default=2)
args = ap.parse_args(); t0 = time.time()
lab = pd.read_csv(f"{args.data}/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv(f"{args.data}/sample_submission.csv", dtype={"id": str})
r = lambda a: rankdata(a) / len(a)
def booster(path):
    """LightGBM text model, stored gzip-compressed in the repo (identical predictions to the uncompressed file)."""
    if os.path.exists(path): return lgb.Booster(model_file=path)
    with gzip.open(path + ".gz", "rt") as f: return lgb.Booster(model_str=f.read())


ld = lambda p, idx: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def frame(idx, part, fam):
    keep = json.load(open(f"{args.feat}/harvest_keep.json"))
    N = pd.concat([ld(f"{args.feat}/factory/{f}.parquet", idx) for f in FAMS + [fam]], axis=1)
    N = N.loc[:, N.columns.str.contains(PAT, regex=True)]
    X = pd.concat([ld(f"{args.feat}/feat_{part}.parquet", idx), ld(f"{args.feat}/magic_b_{part}.parquet", idx),
                   ld(f"{args.feat}/harvest_{part}.parquet", idx)[keep], N.loc[:, ~N.columns.duplicated()]], axis=1)
    return X


def pair_arm(tag, K, fam):
    X = frame(lab.index, "train", fam); X = X.loc[:, ~X.columns.duplicated()]; Xt = frame(sub["id"], "test", fam)[X.columns]
    ALL = np.vstack([X.to_numpy(np.float32), Xt.to_numpy(np.float32)]); ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL)
    n = len(lab); pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]; te = np.zeros(len(sub))
    for k in range(5):
        d = f"{args.ckpt}/{tag}_ck/fold{k}"; top = np.load(f"{d}/top.npy")
        R = np.column_stack([(rankdata(ALL[:, j]) / len(ALL)).astype(np.float32) for j in top])
        # base columns enter unfilled (NaN), exactly as in training
        Z = np.hstack([Xt.to_numpy(np.float32), np.column_stack([R[n:, i] - R[n:, j] for i, j in pairs] + [R[n:, i] * R[n:, j] for i, j in pairs]).astype(np.float32)])
        te += np.mean([booster(f"{d}/seed{s}.txt").predict(Z, num_threads=args.threads) for s in range(3)], 0) / 5
    return te


def inter_arm(tag="inter_gnjt162_ck", seeds=2):
    load = lambda p: ld(p, sub["id"])
    POOL_T = pd.concat([load(f"{args.feat}/feat_test.parquet"), load(f"{args.feat}/raw_test.parquet"), load(f"{args.feat}/factory/combo10gnjx_test.parquet")], axis=1)
    te = np.zeros(len(sub))
    for k in range(5):
        d = f"{args.ckpt}/{tag}/fold{k}"; top = json.load(open(f"{d}/top.json")); z = np.load(f"{d}/inter.npz"); med, eps = z["med"], z["eps"]
        X = POOL_T[top].to_numpy(); X = np.where(np.isnan(X), med, X).astype(np.float32)
        def op(o, i, j):
            a, b = X[:, i], X[:, j]
            return {"mul": lambda: a * b, "add": lambda: a + b, "sub": lambda: a - b, "div": lambda: a / (np.abs(b) + eps[j]) * np.sign(b + 1e-12),
                    "rdiv": lambda: b / (np.abs(a) + eps[i]) * np.sign(a + 1e-12), "mulb2": lambda: a * b * b, "mula2": lambda: a * a * b}[o]()
        for s in range(seeds):
            keep = json.load(open(f"{d}/keep_seed{s}.json")); cols = []
            for c in keep:
                if isinstance(c, int): cols.append(X[:, c])
                elif c.startswith("sq__"): cols.append(X[:, int(c.split("__")[1])] ** 2)
                else: o, i, j = c.split("__"); cols.append(op(o, int(i), int(j)))
            M = np.clip(np.stack(cols, 1), -1e30, 1e30)
            te += booster(f"{d}/top800_seed{s}.txt").predict(M, num_threads=args.threads) / (5 * seeds)
    return te


arms = []
for tag, K, fam in PAIR_ARMS:
    arms.append(pair_arm(tag, K, fam)); print(f"{tag}: done {time.time() - t0:.0f}s", flush=True)
pa = r(sum(r(a) for a in arms))
i2 = inter_arm(); print(f"interaction arm: done {time.time() - t0:.0f}s", flush=True)
pred = r(2 * pa + r(i2))
out = pd.DataFrame({"id": sub["id"], "break_probability": pred})
assert out.shape == (len(sub), 2) and np.isfinite(out.break_probability).all() and out.break_probability.between(0, 1).all()
out.to_csv(args.out, index=False); print(f"wrote {args.out} ({len(out)} rows) in {time.time() - t0:.0f}s")
