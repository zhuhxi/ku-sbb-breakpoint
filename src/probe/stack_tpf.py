"""TabPFN as second-level model over the blendK arm groups (OOF ranks -> logit) + series context; outer 5-fold CV (seed 11)."""
import numpy as np, pandas as pd, glob, os
from scipy.stats import rankdata
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
L = pd.read_csv("data/train_labels.csv", dtype={"id": str}); y = L.label.to_numpy(); r = lambda a: rankdata(a) / len(a)
I = np.load("outputs/brute/oof_lgb_inter_gnjt162.npy"); ld = lambda t: np.load(f"outputs/arms/{t}_oof.npy")
G = {"pa": r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")))), "inter": r(I[2]),
     "pv": r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_prev*.npz")))),
     "ca": r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*rctx*_oof.npy")) if "noTTA" not in f and "mt" not in f and "raw" not in f)),
     "ma": r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*mt*_oof.npy")) if not any(x in f for x in ("rctx", "mtf", "mts", "mtb2")))),
     "rm": r(ld("tpf60rctxmt")), "rw": r(sum(r(ld(t)) for t in ("tpf60raw","tpf70raw2","tpf80raw1","tpf65raw3","tpf100raw4"))), "cr": r(ld("tpf60rctxraw")),
     "sd": r(sum(r(ld(t)) for t in ("tpf60sideraw","tpf70sideraw2","tpf65sideraw3","tpf80sideraw1","tpf50sideraw4","tpf40sideraw7"))),
     "sw": r(sum(r(ld(t)) for t in ("tpf40siderw5","tpf50siderw1","tpf60siderw0","tpf65siderw3","tpf70siderw2","tpf80siderw4"))), "mf": r(sum(r(ld(t)) for t in ("tpf60sidemf1_0","tpf70sidemf1_2","tpf50sidemf1_1","tpf65sidemf1_3","tpf40sidemf1_5","tpf80sidemf1_4","tpf55sidemf1_6"))), "nu": r(sum(r(ld(t)) for t in ("tpf60sidemnr_0","tpf70sidemnr_2","tpf50sidemnr_1","tpf65sidemnr_3","tpf40sidemnr_5","tpf80sidemnr_4","tpf60sidemn_0","tpf70sidemn_2","tpf60sidemn2_0"))), "cb": r(sum(r(ld(t)) for t in ("tpf60sidemnrcb_0","tpf70sidemnrcb_2","tpf50sidemnrcb_1","tpf65sidemnrcb_3","tpf40sidemnrcb_5","tpf80sidemnrcb_4","tpf55sidemnrcb_6","tpf75sidemnrcb_7"))), "tf": r(sum(r(ld(t)) for t in ("tpf70sidet_2","tpf55sidet_6","tpf60sidet_0","tpf50sidet_1","tpf65sidet_3"))), "c2": r(sum(r(ld(t)) for t in ("tpf70sidec2t_2","tpf55sidec2t_6","tpf60sidec2t_0","tpf50sidec2t_1","tpf65sidec2t_3"))), "cm": r(sum(r(ld(t)) for t in ("tpf70sidec2m_2","tpf55sidec2m_6","tpf60sidec2m_0","tpf70sidemc2m_2"))), "cv": r(sum(r(ld(t)) for t in ("tpf70sidec2v_2","tpf55sidec2v_6","tpf60sidec2v_0","tpf50sidec2v_1")))}
W = dict(pa=2, inter=1, pv=2, ca=6, ma=9, rm=3, rw=9, cr=3, sd=9, sw=6, mf=12, nu=24, cb=96, tf=96, c2=72, cm=48, cv=192); K = r(sum(W[k] * G[k] for k in W))
A = pd.read_parquet("outputs/factory/semiauto_bag6w.parquet").reindex(L.id); mb = pd.read_parquet("outputs/magic_b_train.parquet").reindex(L.id)
S = pd.read_parquet("data/train_series.parquet"); g = S.groupby("id"); n = g.size().reindex(L.id).to_numpy(); n0 = g["period"].apply(lambda v: (v == 0).sum()).reindex(L.id).to_numpy()
C = np.column_stack([np.log(n), n0 / n] + [A[c].to_numpy() for c in A if c.startswith("saf_family_")] + [A["saf_kind_None"].to_numpy(), A["saf_ntype_t"].to_numpy(), mb["mb_rescaled"].fillna(0).to_numpy()])
if os.environ.get("CTXEXTRA"):
    for f in os.environ["CTXEXTRA"].split(","):
        E_ = pd.read_parquet(f).reindex(L.id); C = np.column_stack([C, E_[[c for c in E_ if "logit" in c or "aux" in c]].to_numpy(np.float32)])
    print("ctx cols", C.shape[1], flush=True)
Z = logit(np.clip(np.column_stack(list(G.values())), 1e-4, 1 - 1e-4)); Zk = logit(np.clip(K, 1e-4, 1 - 1e-4))[:, None]
cv = list(StratifiedKFold(5, shuffle=True, random_state=int(os.environ.get("OSEED", 11))).split(y, y)); rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(300)]
print(f"blendK {roc_auc_score(y, K):.4f} oseed {os.environ.get('OSEED', 11)} fold0 size {len(cv[0][1])} first {cv[0][1][:3]}", flush=True)
SA = np.column_stack([A[c].to_numpy() for c in A if c.startswith("saf_")]); NE = int(os.environ.get("NE", 4))
import feats; FX, _, _ = feats.load(); FX = FX.reindex(L.id)
imp = pd.read_csv("outputs/stack_topfeat.csv").col.tolist() if os.path.exists("outputs/stack_topfeat.csv") else []
TF = FX[imp[:int(os.environ.get("NTOP", 20))]].to_numpy(np.float32) if imp else np.zeros((len(y), 0))
ARMS = [np.load(f) for f in os.environ.get("XARMS", "").split(",") if f] + [np.load(f)["oof"] for f in sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")) + sorted(glob.glob("outputs/factory/confirm_prev*.npz"))] + [np.load(f) for f in sorted(glob.glob("outputs/arms/tpf*_oof.npy")) if "noTTA" not in f]
ZA = logit(np.clip(np.column_stack([r(a) for a in ARMS]), 1e-4, 1 - 1e-4)); print("arms", ZA.shape, flush=True)
CB = pd.read_parquet("outputs/factory/chronos_2.parquet").reindex(L.id).to_numpy(np.float32)
TFE = pd.read_parquet("outputs/factory/seqembtf.parquet").reindex(L.id); TFE = TFE[[c for c in TFE if "logit" in c or "aux" in c]].to_numpy(np.float32)
V = {"arms+ctx+cb+tf": np.column_stack([ZA, Zk, C, CB, TFE]), "arms+ctx+cb": np.column_stack([ZA, Zk, C, CB]), "arms+ctx+top": np.column_stack([ZA, Zk, C, TF]), "arms+ctx": np.column_stack([ZA, Zk, C]), "ctx+top": np.column_stack([Z, Zk, C, TF]),"ctx": np.column_stack([Z, Zk, C]), "ctx+saf": np.column_stack([Z, Zk, C, SA]), "ctx noK": np.column_stack([Z, C])}
for nm in os.environ.get("VARS", "ctx").split(","):
    X = V[nm]
    o = np.zeros(len(y))
    for tr, va in cv:
        for sd in range(int(os.environ.get("NSEED", 1))): o[va] += rankdata(TabPFNClassifier(device="cuda", n_estimators=NE, ignore_pretraining_limits=True, random_state=sd).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1])
    for nm2, s in ((nm, o), (nm + " avgK", r(o) + K), (nm + " 2:1", 2 * r(o) + K), (nm + " 3:1", 3 * r(o) + K)):
        d = [roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], K[i]) for i in B]
        print(f"{nm2:22s} {roc_auc_score(y, s):.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] folds {sum(roc_auc_score(y[va], s[va]) > roc_auc_score(y[va], K[va]) for _, va in cv)}/5", flush=True)
