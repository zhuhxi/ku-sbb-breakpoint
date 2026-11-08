"""Zero-shot scores (+ pooled 3840-d embeddings, PCA32) of a fine-tuned Chronos-2 classifier on real train+test, fwd and reversed tensors."""
import sys, os, numpy as np, pandas as pd, torch
sys.argv_ = sys.argv[:]; CK, TAG = sys.argv[1], sys.argv[2]; os.environ.setdefault("WIN", "512"); sys.argv = [sys.argv[0], "x", "0", "0", "x"]
exec(open("probe/tf_ft.py").read().split("net = C2Cls().to(dev)")[0]); net = C2Cls().to(dev); net.load_state_dict(torch.load(CK)); net.eval()
from sklearn.decomposition import PCA
def run(T):
    X = win(T); S, E = [], []; global AX; AX = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 512):
            e = net.emb(torch.tensor(X[i:i + 512], device=dev)); S.append(net.h(e).squeeze(-1).float().cpu().numpy()); E.append(e.float().cpu().numpy()); AX.append(net.aux(e).float().cpu().numpy())
    return np.concatenate(S), np.concatenate(E), np.concatenate(AX)
sf, ef, af = run(np.concatenate([np.load("outputs/arms/seq_train.npy"), np.load("outputs/arms/seq_test.npy")]))
sr, er, ar = run(np.concatenate([np.load("outputs/arms/seq_train_rev.npy"), np.load("outputs/arms/seq_test_rev.npy")]))
np.save(f"outputs/arms/zs{TAG}_oof.npy", sf[:24000]); np.save(f"outputs/arms/zs{TAG}_test.npy", sf[24000:]); np.save(f"outputs/arms/zs{TAG}rev_oof.npy", sr[:24000]); np.save(f"outputs/arms/zs{TAG}rev_test.npy", sr[24000:])
mu, sd = ef.mean(0), ef.std(0) + 1e-6; pca = PCA(32, random_state=0).fit((ef - mu) / sd)
ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
for E, s, AXv, out in ((ef, sf, af, f"outputs/factory/seqemb{TAG}.parquet"), (er, sr, ar, f"/mnt/disk2/zhuhx/kagglecompetition/ku_sbb_rev/outputs/factory/seqemb{TAG}.parquet")):
    D = pd.DataFrame(pca.transform((E - mu) / sd), index=pd.Index(ids, name="id"), columns=[f"se_{TAG}_pc{i}" for i in range(32)]); D[f"se_{TAG}_logit"] = s
    for i_ in range(AXv.shape[1]): D[f"se_{TAG}_aux{i_}"] = AXv[:, i_]
    D.to_parquet(out)
print("done")
