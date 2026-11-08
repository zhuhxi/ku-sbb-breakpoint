"""Blend check of extra arms against blendI (.8674)."""
import sys, numpy as np, pandas as pd, glob
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
y=pd.read_csv("data/train_labels.csv").label.to_numpy(); r=lambda a: rankdata(a)/len(a); I=np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
pa=r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")))); pv=r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_prev*.npz"))))
ca=r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*rctx*_oof.npy")) if "noTTA" not in f and "mt" not in f and "raw" not in f)); rm=r(np.load("outputs/arms/tpf60rctxmt_oof.npy"))
mf=[f for f in sorted(glob.glob("outputs/arms/tpf*mt*_oof.npy")) if not any(x in f for x in ("rctx","mtf","mts","mtb2"))]; ma=r(sum(r(np.load(f)) for f in mf))
rw=r(sum(r(np.load(f"outputs/arms/{t}_oof.npy")) for t in ("tpf60raw","tpf70raw2","tpf80raw1","tpf65raw3","tpf100raw4"))); cr=r(np.load("outputs/arms/tpf60rctxraw_oof.npy"))
core=2*pa+r(I[2])+2*pv+6*ca+9*ma+3*rm+9*rw+3*cr; Ib=r(core); print("blendI", round(roc_auc_score(y,Ib),4))
ex=r(sum(r(np.load(f)) for f in sys.argv[1:])); print(len(sys.argv)-1, "extra avg", round(roc_auc_score(y,ex),4), "corr w/ raw group", round(np.corrcoef(ex,rw)[0,1],3))
rng=np.random.default_rng(0); B=[rng.integers(0,len(y),len(y)) for _ in range(300)]; cv=list(StratifiedKFold(5,shuffle=True,random_state=7).split(y,y))
for w in (3,6,9,12):
    s=r(core+w*ex); d=[roc_auc_score(y[i],s[i])-roc_auc_score(y[i],Ib[i]) for i in B]
    print(f"I + extra x{w}", round(roc_auc_score(y,s),4), f"CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] folds {sum(roc_auc_score(y[va],s[va])>roc_auc_score(y[va],Ib[va]) for _,va in cv)}/5")
