"""MLP on each fold's top-K gain features (ranks -> normal scores), fold-clean selection; seeds averaged."""
import sys, time, numpy as np, torch, torch.nn as nn, lightgbm as lgb
from scipy.stats import rankdata, norm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "probe"); import feats
K = int(sys.argv[1]); NS = int(sys.argv[2]); H = int(sys.argv[3]); DR = float(sys.argv[4]); t0 = time.time(); dev = "cuda"
X, Xt, y = feats.load(); Xn, XT = X.to_numpy(np.float32), Xt.to_numpy(np.float32); ALL = np.vstack([Xn, XT]); ALL = np.where(np.isnan(ALL), np.nanmedian(ALL, 0), ALL)
Q = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_estimators=2000, verbose=-1, n_jobs=16)
oof = np.zeros(len(y)); te = np.zeros(len(XT))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xn, y)):
    a0, b0 = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=k)
    m0 = lgb.LGBMClassifier(**Q, random_state=0).fit(Xn[a0], y[a0], eval_set=[(Xn[b0], y[b0])], callbacks=[lgb.early_stopping(50, verbose=False)])
    top = np.argsort(-m0.booster_.feature_importance("gain"))[:K]
    R = norm.ppf(np.column_stack([rankdata(ALL[:, j]) / (len(ALL) + 1) for j in top])).astype(np.float32); Rt = torch.tensor(R, device=dev)
    for s in range(NS):
        torch.manual_seed(s); a, b = train_test_split(tr, test_size=.1, stratify=y[tr], random_state=s)
        m = nn.Sequential(nn.Linear(K, H), nn.BatchNorm1d(H), nn.SiLU(), nn.Dropout(DR), nn.Linear(H, H), nn.BatchNorm1d(H), nn.SiLU(), nn.Dropout(DR), nn.Linear(H, 1)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-2); ya = torch.tensor(y[a], dtype=torch.float32, device=dev); best, bst, bad = 0, None, 0
        for ep in range(200):
            m.train(); perm = torch.tensor(np.random.default_rng(ep + 1000 * s).permutation(a), device=dev)
            for i in range(0, len(a), 256):
                j = perm[i:i+256]; opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(m(Rt[j]).ravel(), torch.tensor(y, dtype=torch.float32, device=dev)[j]).backward(); opt.step()
            m.eval()
            with torch.no_grad(): sc = roc_auc_score(y[b], m(Rt[b]).ravel().cpu().numpy())
            if sc > best: best, bad, bst = sc, 0, {kk: v.clone() for kk, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 15: break
        m.load_state_dict(bst); m.eval()
        with torch.no_grad(): oof[va] += torch.sigmoid(m(Rt[va])).ravel().cpu().numpy() / NS; te += torch.sigmoid(m(Rt[len(y):])).ravel().cpu().numpy() / NS / 5
    print(f"fold {k} {roc_auc_score(y[va], oof[va]):.4f} {time.time()-t0:.0f}s", flush=True)
print(f"mlptop K{K} OOF {roc_auc_score(y, oof):.4f}"); np.save(f"outputs/arms/mlptop{K}_oof.npy", oof); np.save(f"outputs/arms/mlptop{K}_test.npy", te)
