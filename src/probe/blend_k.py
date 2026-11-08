"""Blend check of extra arms against blendK (.8684): extra arms added into the siderw group (weight 24 on the group)."""
import sys, numpy as np, pandas as pd, glob
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
y=pd.read_csv("data/train_labels.csv").label.to_numpy(); r=lambda a: rankdata(a)/len(a); I=np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
pa=r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")))); pv=r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_prev*.npz"))))
ca=r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*rctx*_oof.npy")) if "noTTA" not in f and "mt" not in f and "raw" not in f)); rm=r(np.load("outputs/arms/tpf60rctxmt_oof.npy"))
mf=[f for f in sorted(glob.glob("outputs/arms/tpf*mt*_oof.npy")) if not any(x in f for x in ("rctx","mtf","mts","mtb2"))]; ma=r(sum(r(np.load(f)) for f in mf))
rw=r(sum(r(np.load(f"outputs/arms/{t}_oof.npy")) for t in ("tpf60raw","tpf70raw2","tpf80raw1","tpf65raw3","tpf100raw4"))); cr=r(np.load("outputs/arms/tpf60rctxraw_oof.npy"))
sd=r(sum(r(np.load(f"outputs/arms/{t}_oof.npy")) for t in ("tpf60sideraw","tpf70sideraw2","tpf65sideraw3","tpf80sideraw1","tpf50sideraw4","tpf40sideraw7")))
w6=[f"outputs/arms/{t}_oof.npy" for t in ("tpf40siderw5","tpf50siderw1","tpf60siderw0","tpf65siderw3","tpf70siderw2","tpf80siderw4")]
core=2*pa+r(I[2])+2*pv+6*ca+9*ma+3*rm+9*rw+3*cr+9*sd; K=r(core+24*r(sum(r(np.load(f)) for f in w6)))
rng=np.random.default_rng(0); B=[rng.integers(0,len(y),len(y)) for _ in range(300)]; cv=list(StratifiedKFold(5,shuffle=True,random_state=7).split(y,y))
def rep(nm,s):
    d=[roc_auc_score(y[i],s[i])-roc_auc_score(y[i],K[i]) for i in B]; print(f"{nm:30s}", round(roc_auc_score(y,s),4), f"CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] folds {sum(roc_auc_score(y[va],s[va])>roc_auc_score(y[va],K[va]) for _,va in cv)}/5")
print("blendK", round(roc_auc_score(y,K),4)); ex=sys.argv[1:]
rep("group + extra", r(core+24*r(sum(r(np.load(f)) for f in w6+ex))))
rep("group with extra replacing w6[2]", r(core+24*r(sum(r(np.load(f)) for f in [f for f in w6 if "tpf60siderw0" not in f]+ex))))
