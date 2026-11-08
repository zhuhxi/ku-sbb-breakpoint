import numpy as np, pandas as pd, glob, sys
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
y=pd.read_csv("data/train_labels.csv").label.to_numpy(); r=lambda a: rankdata(a)/len(a); I=np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
pa=r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")))); base=r(2*pa+r(I[2]))
tf=[f for f in sorted(glob.glob("outputs/arms/tpf*_oof.npy")) if not any(x in f for x in ("tpf30_","tpf150_","ne16","c8e2","c4e1","ne1"))]; ta=r(sum(r(np.load(f)) for f in tf))
pr=sorted(glob.glob("outputs/factory/confirm_prev*.npz")); pv=r(sum(r(np.load(f)["oof"]) for f in pr)); rv=r(np.load("outputs/factory/confirm_rev919.npz")["oof"])
print(len(pr), "prev arms avg", round(roc_auc_score(y,pv),4))
rng=np.random.default_rng(0); B=[rng.integers(0,len(y),len(y)) for _ in range(300)]; cv=list(StratifiedKFold(5,shuffle=True,random_state=7).split(y,y))
def rep(nm,s,b):
    d=[roc_auc_score(y[i],s[i])-roc_auc_score(y[i],b[i]) for i in B]; print(f"{nm:40s}", round(roc_auc_score(y,s),4), f"{roc_auc_score(y,s)-roc_auc_score(y,b):+.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] folds {sum(roc_auc_score(y[va],s[va])>roc_auc_score(y[va],b[va]) for _,va in cv)}/5")
for w in (1,2,4): rep(f"LGB blend + prev x{w}", r(2*pa+r(I[2])+w*pv), base)
rep("prev x2 + inter (no old pairs)", r(2*pv+r(I[2])), base)
rep("LGB + prev x2 + rev919", r(2*pa+r(I[2])+2*pv+rv), base)
bF=r(2*pa+r(I[2])+6*ta); print("blendF", round(roc_auc_score(y,bF),4))
for w in (2,4): rep(f"blendF + prev x{w}", r(2*pa+r(I[2])+6*ta+w*pv), bF)
