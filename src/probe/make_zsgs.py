"""Label-free zero-shot arm 'zsgs' used by the blendRFS stacker: mean rank of the zero-shot break logits of three CNNs pretrained
on genS fakes (ckpt/seq/seqpre_h_shape25d7, seqpre_h_shape25b, seqpre_gsr), each on the original and the time-reversed tensors.
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=probe python probe/make_zsgs.py      (run from the project root; checkpoints in outputs/arms/)"""
import sys, numpy as np, torch
sys.argv = ["x", "0"]; import seq_pretrain as sp
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
y = sp.yr; Xa = torch.cat([sp.Xr, torch.tensor(np.load("outputs/arms/seq_test.npy"))])
Xr = torch.cat([torch.tensor(np.load("outputs/arms/seq_train_rev.npy")), torch.tensor(np.load("outputs/arms/seq_test_rev.npy"))])
tr = np.zeros(24000); te = np.zeros(4000)
for nm in ("h_shape25d7", "h_shape25b", "gsr"):
    m = sp.Net().to(sp.dev); m.load_state_dict(torch.load(f"outputs/arms/seqpre_{nm}_mt_c64.pt"), strict=False)
    for X in (Xa, Xr):
        p = sp.predict(m, X); tr += rankdata(p[:24000]) / 24000; te += rankdata(p[24000:]) / 4000; print(nm, round(roc_auc_score(y, p[:24000]), 4))
print("avg of 6", round(roc_auc_score(y, tr), 4)); np.save("outputs/arms/zsgs_oof.npy", tr); np.save("outputs/arms/zsgs_test.npy", te)
