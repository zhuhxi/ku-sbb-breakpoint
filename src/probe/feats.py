"""Load the production feature set of combo10gnjbag6 (train X, test Xt in sample-submission order)."""
import json, numpy as np, pandas as pd
FAMS = ["st_RES10", "st_ARES10", "fe", "fe2", "ar2", "cost", "genA", "genB", "gnoise", "gjlr", "semiauto_bag6"]
PAT = r"^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_|^saf_|^se_|^cb_|^d_|^l_|^cbb_|^cbl_|^cbr_|^fc_|^tfm_|^tfz_|^tfh_|^fma_"
lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
y = lab.to_numpy()
ld = lambda p, idx: pd.read_parquet(p).reindex(idx).replace([np.inf, -np.inf], np.nan).astype(np.float32)
def frame(idx, part, fams=FAMS):
    keep = json.load(open("outputs/harvest_keep.json"))
    N = pd.concat([ld(f"outputs/factory/{f}.parquet", idx) for f in fams], axis=1)
    N = N.loc[:, N.columns.str.contains(PAT, regex=True)]
    return pd.concat([ld(f"outputs/feat_{part}.parquet", idx), ld(f"outputs/magic_b_{part}.parquet", idx),
                      ld(f"outputs/harvest_{part}.parquet", idx)[keep], N.loc[:, ~N.columns.duplicated()]], axis=1)
def load():
    X, Xt = frame(lab.index, "train"), frame(sub["id"], "test")
    X = X.loc[:, ~X.columns.duplicated()]; return X, Xt[X.columns], y
