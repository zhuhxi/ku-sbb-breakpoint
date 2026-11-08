"""blendKS = rank(rank(TabPFN stacker) + blendK). Stacker: TabPFN (NE 8) on logit of the 10 blendK group ranks + blendK + context,
fitted on all 24k OOF rows, applied to the test group ranks. OOF of the same recipe (outer CV seeds 11/12/13): .8687/.8686/.8688."""
import numpy as np, pandas as pd, glob
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy(); sub = pd.read_csv("data/sample_submission.csv", dtype={"id": str})
r = lambda a: rankdata(np.round(a, 12)) / len(a)
I = {"oof": np.load("outputs/brute/oof_lgb_inter_gnjt162.npy"), "test": np.load("outputs/brute/test_lgb_inter_gnjt162.npy")}
pf = sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")); vf = sorted(glob.glob("outputs/factory/confirm_prev*.npz"))
cf = [f for f in sorted(glob.glob("outputs/arms/tpf*rctx*_oof.npy")) if "noTTA" not in f and "mt" not in f and "raw" not in f]
mf = [f for f in sorted(glob.glob("outputs/arms/tpf*mt*_oof.npy")) if not any(x in f for x in ("rctx", "mtf", "mts", "mtb2"))]
A_ = lambda ts: [f"outputs/arms/{t}_oof.npy" for t in ts]
rwf = A_(("tpf60raw", "tpf70raw2", "tpf80raw1", "tpf65raw3", "tpf100raw4")); sdf = A_(("tpf60sideraw", "tpf70sideraw2", "tpf65sideraw3", "tpf80sideraw1", "tpf50sideraw4", "tpf40sideraw7"))
swf = A_(("tpf40siderw5", "tpf50siderw1", "tpf60siderw0", "tpf65siderw3", "tpf70siderw2", "tpf80siderw4"))
import os; FULLT = bool(os.environ.get("FULLT")); print("full-context test preds:", FULLT)
W = dict(pa=2, inter=1, pv=2, ca=6, ma=9, rm=3, rw=9, cr=3, sd=9, sw=6, mf=12, nu=24, cb=96, tf=96); GG = {}
for p in ("oof", "test"):
    tl = lambda f: np.load(f) if p == "oof" else (np.load(f.replace("_oof.npy", "full_f0.npz"))["test"] if FULLT and os.path.exists(f.replace("_oof.npy", "full_f0.npz")) else np.load(f.replace("_oof.npy", "_test.npy"))); g = lambda fs: r(sum(r(tl(f)) for f in fs))
    GG[p] = {"pa": r(sum(r(np.load(f)[p]) for f in pf)), "inter": r(I[p][2]), "pv": r(sum(r(np.load(f)[p]) for f in vf)), "ca": g(cf), "ma": g(mf),
             "rm": r(tl("outputs/arms/tpf60rctxmt_oof.npy")), "rw": g(rwf), "cr": r(tl("outputs/arms/tpf60rctxraw_oof.npy")), "sd": g(sdf), "sw": g(swf), "mf": g(A_(("tpf60sidemf1_0","tpf70sidemf1_2","tpf50sidemf1_1","tpf65sidemf1_3","tpf40sidemf1_5","tpf80sidemf1_4","tpf55sidemf1_6"))), "nu": g(A_(("tpf60sidemnr_0","tpf70sidemnr_2","tpf50sidemnr_1","tpf65sidemnr_3","tpf40sidemnr_5","tpf80sidemnr_4","tpf60sidemn_0","tpf70sidemn_2","tpf60sidemn2_0"))), "cb": g(A_(("tpf60sidemnrcb_0","tpf70sidemnrcb_2","tpf50sidemnrcb_1","tpf65sidemnrcb_3","tpf40sidemnrcb_5","tpf80sidemnrcb_4","tpf55sidemnrcb_6","tpf75sidemnrcb_7"))), "tf": g(A_(("tpf70sidet_2","tpf55sidet_6","tpf60sidet_0","tpf50sidet_1","tpf65sidet_3")))}
def ctx(ids, part):
    A = pd.read_parquet("outputs/factory/semiauto_bag6w.parquet").reindex(ids); mb = pd.read_parquet(f"outputs/magic_b_{part}.parquet").reindex(ids)
    S = pd.read_parquet(f"data/{part}_series.parquet"); g = S.groupby("id"); n = g.size().reindex(ids).to_numpy(); n0 = g["period"].apply(lambda v: (v == 0).sum()).reindex(ids).to_numpy()
    C0 = np.column_stack([np.log(n), n0 / n] + [A[c].to_numpy() for c in A if c.startswith("saf_family_")] + [A["saf_kind_None"].to_numpy(), A["saf_ntype_t"].to_numpy(), mb["mb_rescaled"].fillna(0).to_numpy()])
    for f in [f for f in os.environ.get("CTXEXTRA", "").split(",") if f]:
        E_ = pd.read_parquet(f).reindex(ids); C0 = np.column_stack([C0, E_[[c for c in E_ if "logit" in c or "aux" in c]].to_numpy(np.float32)])
    return C0
K = {p: r(sum(W[k] * GG[p][k] for k in W)) for p in GG}
print("blendK OOF", round(roc_auc_score(y, K["oof"]), 4), "| test corr with submitted blendK", np.corrcoef(K["test"], pd.read_csv("outputs/submission_blendK.csv").break_probability)[0, 1])
import os
ARMS = sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")) + sorted(glob.glob("outputs/factory/confirm_prev*.npz")) + [f for f in sorted(glob.glob("outputs/arms/tpf*_oof.npy")) if "noTTA" not in f]
arm = lambda f, p: np.load(f)[p] if f.endswith(".npz") else (np.load(f) if p == "oof" else (np.load(f.replace("_oof.npy", "full_f0.npz"))["test"] if FULLT and os.path.exists(f.replace("_oof.npy", "full_f0.npz")) else np.load(f.replace("_oof.npy", "_test.npy"))))
ARMS = [f for f in os.environ.get("XARMS", "").split(",") if f] + ARMS; USE_ARMS = bool(os.environ.get("ARMS")); print("arms", len(ARMS), "use", USE_ARMS)
src = lambda p: np.column_stack([r(arm(f, p)) for f in ARMS]) if USE_ARMS else np.column_stack(list(GG[p].values()))
X = {p: np.column_stack([logit(np.clip(src(p), 1e-4, 1 - 1e-4)), logit(np.clip(K[p], 1e-4, 1 - 1e-4)), ctx(L.id if p == "oof" else sub.id, "train" if p == "oof" else "test")]) for p in GG}
st = TabPFNClassifier(device="cuda", n_estimators=8, ignore_pretraining_limits=True, random_state=0).fit(X["oof"], y).predict_proba(X["test"])[:, 1]
SW = float(os.environ.get("SW", 1)); out = r(SW * r(st) + K["test"]); print("stack weight", SW); pd.DataFrame({"id": sub.id, "break_probability": K["test"]}).to_csv(os.environ.get("OUT", "x.csv").replace("S.csv", ".csv").replace("S2.csv", ".csv"), index=False); print("corr(stack, K) test", np.corrcoef(r(st), K["test"])[0, 1])
pd.DataFrame({"id": sub.id, "break_probability": out}).to_csv(os.environ.get("OUT", "outputs/submission_blendKS.csv"), index=False); print("written outputs/submission_blendKS.csv", out.shape)
