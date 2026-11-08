"""Blend check of extra arms (paths) against G2 (2*pairs + inter + 2*rev-pairs + 6*TabPFN + 6*rev-ctx TabPFN) and LGB-only."""
import sys, numpy as np, pandas as pd, glob
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
y=pd.read_csv("data/train_labels.csv").label.to_numpy(); r=lambda a: rankdata(a)/len(a); I=np.load("outputs/brute/oof_lgb_inter_gnjt162.npy")
pa=r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_pair[0-9bs]*.npz")))); pv=r(sum(r(np.load(f)["oof"]) for f in sorted(glob.glob("outputs/factory/confirm_prev*.npz"))))
sk=("tpf30_","tpf150_","ne16","c8e2","c4e1","ne1","rtta","rctx","ps25","psel","tpf60se"); ta=r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*_oof.npy")) if not any(x in f for x in sk)))
ca=r(sum(r(np.load(f)) for f in sorted(glob.glob("outputs/arms/tpf*rctx*_oof.npy")) if "noTTA" not in f))
L=r(2*pa+r(I[2])+2*pv); G=r(2*pa+r(I[2])+2*pv+6*ta+6*ca)
rng=np.random.default_rng(0); B=[rng.integers(0,len(y),len(y)) for _ in range(300)]; cv=list(StratifiedKFold(5,shuffle=True,random_state=7).split(y,y))
def rep(nm,s,b):
    d=[roc_auc_score(y[i],s[i])-roc_auc_score(y[i],b[i]) for i in B]; print(f"{nm:36s}", round(roc_auc_score(y,s),4), f"{roc_auc_score(y,s)-roc_auc_score(y,b):+.4f} CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] folds {sum(roc_auc_score(y[va],s[va])>roc_auc_score(y[va],b[va]) for _,va in cv)}/5")
for p in sys.argv[1:]:
    z=np.load(p); a=r(z["oof"] if hasattr(z,"files") else z); print(p.split("/")[-1], "alone", round(roc_auc_score(y,a),4))
    for w in (1,2,4): rep(f"  LGB + x{w}", r(2*pa+r(I[2])+2*pv+w*a), L)
    for w in (2,4,6): rep(f"  G + x{w}", r(2*pa+r(I[2])+2*pv+6*ta+6*ca+w*a), G)
