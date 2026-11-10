"""Negation test-time augmentation for a fine-tuned two-variate Chronos-2 classifier: logits on -x (z and AR-residual channels negated),
original and reversed tensors. Writes zs{TAG}neg_{oof,test}.npy and zs{TAG}negrev_*.   python probe/c2_tta.py CKPT TAG"""
import sys, os, numpy as np, pandas as pd, torch
CK, TAG = sys.argv[1], sys.argv[2]; os.environ.setdefault("WIN", "512"); sys.argv = [sys.argv[0], "x", "0", "0", "x"]
exec(open("probe/c2mv_ft.py").read().split("net = C2Cls().to(dev)")[0]); net = C2Cls().to(dev); net.load_state_dict(torch.load(CK), strict=False); net.eval()
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
def run(T):
    X = win(T); X[:, :2] *= -1; S = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 256): S.append(net(torch.tensor(X[i:i + 256], device=dev)).float().cpu().numpy())
    return np.concatenate(S)
y = pd.read_csv("data/train_labels.csv").label.values; r = lambda a: rankdata(a) / len(a)
for sfx, a, b in (("neg", "seq_train", "seq_test"), ("negrev", "seq_train_rev", "seq_test_rev")):
    s = run(np.concatenate([np.load(f"outputs/arms/{a}.npy"), np.load(f"outputs/arms/{b}.npy")]))
    np.save(f"outputs/arms/zs{TAG}{sfx}_oof.npy", s[:24000]); np.save(f"outputs/arms/zs{TAG}{sfx}_test.npy", s[24000:]); print(sfx, round(roc_auc_score(y, s[:24000]), 4), flush=True)
V = [np.load(f"outputs/arms/zs{TAG}{k}_oof.npy") for k in ("", "rev", "neg", "negrev")]
print("fwd+rev", round(roc_auc_score(y, r(V[0]) + r(V[1])), 4), "| all 4 views", round(roc_auc_score(y, sum(r(v) for v in V)), 4))
