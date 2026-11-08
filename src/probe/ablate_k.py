"""Leave-one-group-out ablation of blendK (OOF AUC change, paired bootstrap 95% CI, folds down)."""
import numpy as np, pandas as pd, glob
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
y=pd.read_csv("data/train_labels.csv").label.to_numpy(); r=lambda a: rankdata(a)/len(a); ld=lambda t: np.load(f"outputs/arms/{t}_oof.npy")
G={"pa": r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")))), "inter": r(np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")[2]),
   "pv": r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_prev*.npz")))),
   "ca": r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*rctx*_oof.npy")) if "noTTA" not in f and "mt" not in f and "raw" not in f)),
   "ma": r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*mt*_oof.npy")) if not any(x in f for x in ("rctx","mtf","mts","mtb2")))),
   "rm": r(ld("tpf60rctxmt")), "rw": r(sum(r(ld(t)) for t in ("tpf60raw","tpf70raw2","tpf80raw1","tpf65raw3","tpf100raw4"))), "cr": r(ld("tpf60rctxraw")),
   "sd": r(sum(r(ld(t)) for t in ("tpf60sideraw","tpf70sideraw2","tpf65sideraw3","tpf80sideraw1","tpf50sideraw4","tpf40sideraw7"))),
   "sw": r(sum(r(ld(t)) for t in ("tpf40siderw5","tpf50siderw1","tpf60siderw0","tpf65siderw3","tpf70siderw2","tpf80siderw4")))}
W=dict(pa=2,inter=1,pv=2,ca=6,ma=9,rm=3,rw=9,cr=3,sd=9,sw=24); bl=lambda ks: sum(W[k]*G[k] for k in ks); K=bl(W)
rng=np.random.default_rng(0); B=[rng.integers(0,len(y),len(y)) for _ in range(300)]; cv=list(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y))
print("blendK", round(roc_auc_score(y,K),4))
sets={"- pair arms":["pa"],"- interaction":["inter"],"- reversal pair":["pv"],"- all LightGBM":["pa","inter","pv"],"- all reversal (pv,ca,rm,cr,sd,sw)":["pv","ca","rm","cr","sd","sw"],
      "- CNN-feature groups (ma,rm,sw)":["ma","rm","sw"],"- all TabPFN":["ca","ma","rm","rw","cr","sd","sw"]}
for nm,drop in sets.items():
    s=bl([k for k in W if k not in drop]); d=[roc_auc_score(y[i],s[i])-roc_auc_score(y[i],K[i]) for i in B]
    print(f"{nm:38s} {roc_auc_score(y,s):.4f} {roc_auc_score(y,s)-roc_auc_score(y,K):+.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] folds down {sum(roc_auc_score(y[va],s[va])<roc_auc_score(y[va],K[va]) for _,va in cv)}/5")
