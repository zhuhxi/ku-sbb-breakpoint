"""Blend check of extra TabPFN-like arms against H (blendH2, .8669): either added as a separate group or merged into the MT group."""
import sys, numpy as np, pandas as pd, glob
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
y=pd.read_csv("data/train_labels.csv").label.to_numpy(); r=lambda a: rankdata(a)/len(a); I=np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
pa=r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")))); pv=r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_prev*.npz"))))
ca=r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*rctx*_oof.npy")) if "noTTA" not in f and "mt" not in f)); rm=r(np.load("outputs/arms/tpf60rctxmt_oof.npy"))
mf=[f for f in sorted(glob.glob("outputs/arms/tpf*mt*_oof.npy")) if not any(x in f for x in ("rctx","mtf","mts","mtb2"))]; ma=r(sum(r(np.load(f)) for f in mf))
H=r(2*pa+r(I[2])+2*pv+6*ca+9*ma+3*rm); print("H", round(roc_auc_score(y,H),4))
rng=np.random.default_rng(0); B=[rng.integers(0,len(y),len(y)) for _ in range(300)]; cv=list(StratifiedKFold(5,shuffle=True,random_state=7).split(y,y))
ex=r(sum(r(np.load(f)) for f in sys.argv[1:])); print(len(sys.argv)-1, "extra arms avg", round(roc_auc_score(y,ex),4), "corr w/ MT group", round(np.corrcoef(ex,ma)[0,1],3))
for nm,s in [(f"H + extra x{w}", r(2*pa+r(I[2])+2*pv+6*ca+9*ma+3*rm+w*ex)) for w in (3,6,9,12)] + [("replace MT group by extra", r(2*pa+r(I[2])+2*pv+6*ca+9*ex+3*rm)), ("MT x6 + extra x6", r(2*pa+r(I[2])+2*pv+6*ca+6*ma+3*rm+6*ex))]:
    d=[roc_auc_score(y[i],s[i])-roc_auc_score(y[i],H[i]) for i in B]; print(f"{nm:28s}", round(roc_auc_score(y,s),4), f"CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] folds {sum(roc_auc_score(y[va],s[va])>roc_auc_score(y[va],H[va]) for _,va in cv)}/5")
