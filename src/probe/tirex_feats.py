"""TiRex (NX-AI/TiRex, xLSTM) version of chronos_feats.py (prefix tx_).  python probe/tirex_feats.py N_OR_ALL OUT
Original doc: Chronos-Bolt zero-shot forecast features (public pretrained model, local GPU): forecast the first H post-boundary points from the
pre-boundary context and compare with the actual values; the same for a back-test inside the pre segment (last H pre points from
earlier pre points) to self-normalise. No labels used.
    CUDA_VISIBLE_DEVICES=0 python probe/chronos_feats.py MODEL N_OR_ALL OUT"""
import sys, time, numpy as np, pandas as pd, torch
from tirex import load_model
NS, OUT = sys.argv[1], sys.argv[2]; H = int(__import__("os").environ.get("HOR", 64)); CTX = int(__import__("os").environ.get("CTXL", 512)); t0 = time.time()
pipe = load_model("NX-AI/TiRex", device="cuda")
S = pd.concat([pd.read_parquet("data/train_series.parquet"), pd.read_parquet("data/test_series.parquet")]); S["id"] = S["id"].astype(str)
S = S.sort_values(["id", "time"], kind="stable")
if __import__("os").environ.get("REVSEQ"):   # time-reversed series (boundary at n - n0)
    n = S.groupby("id")["time"].transform("max"); S = S.assign(time=n - S["time"], period=1 - S["period"]).sort_values(["id", "time"], kind="stable")
ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
if NS != "all": ids = ids[:int(NS)]
G = {k: (g["value"].to_numpy(np.float32), int((g["period"].to_numpy() == 0).sum())) for k, g in S[S["id"].isin(set(ids))].groupby("id", sort=False)}
if __import__("os").environ.get("ARRES"):   # AR(10) residuals fitted on the pre segment (whitened series)
    def _res(v, n0, q=10):
        v = v.astype(float); Zl = np.column_stack([np.ones(len(v) - q)] + [v[q - j - 1:len(v) - j - 1] for j in range(q)])
        c = np.linalg.lstsq(Zl[:max(n0 - q, q + 2)], v[q:max(n0, 2 * q + 2)], rcond=None)[0]; e = v[q:] - Zl @ c
        return np.r_[np.zeros(q), e].astype(np.float32)
    G = {k: (_res(v, n0), n0) for k, (v, n0) in G.items()}
QL = [0.1, 0.5, 0.9]
def fc(contexts, h):
    out = []
    for i in range(0, len(contexts), 512):
        cs = contexts[i:i + 512]; L = max(len(c) for c in cs); X = np.full((len(cs), L), np.nan, np.float32)
        for j, c in enumerate(cs): X[j, L - len(c):] = c   # left NaN padding
        q, _ = pipe.forecast(context=torch.tensor(X), prediction_length=h)   # 9 quantiles .1 .. .9
        out.append(q[..., [0, 4, 8]].float().cpu().numpy())
    return np.concatenate(out)   # (n, h, 3)
def feats(Q, A, sc):
    lo, md, hi = Q[..., 0], Q[..., 1], Q[..., 2]; w = (hi - lo) + 1e-6 * sc[:, None]; d = {}
    for m in (8, 16, 32, 64):
        e = (A[:, :m] - md[:, :m]); d[f"e{m}"] = np.nanmean(np.abs(e) / w[:, :m], 1); d[f"b{m}"] = np.nanmean(e, 1) / (np.nanmean(w[:, :m], 1) + 1e-9)
        d[f"out{m}"] = np.nanmean((A[:, :m] < lo[:, :m]) | (A[:, :m] > hi[:, :m]), 1)
        d[f"sd{m}"] = np.log((np.nanstd(A[:, :m], 1) + 1e-6) / (np.nanmean(w[:, :m], 1) + 1e-6))
    return d
post_ctx, post_act, pre_ctx, pre_act, sc = [], [], [], [], []
for k in ids:
    v, n0 = G[k]; s = v[:n0].std() + 1e-6; sc.append(s)
    LT = int(__import__("os").environ.get("LATE", 0)); e0 = min(n0 + LT, len(v) - 8)   # LATE: context also includes the first LT post points
    post_ctx.append(v[max(0, e0 - CTX):e0]); a = np.full(H, np.nan, np.float32); seg = v[e0:e0 + H]; a[:len(seg)] = seg; post_act.append(a)
    hb = min(H, n0 // 3); pre_ctx.append(v[max(0, n0 - hb - CTX):n0 - hb]); b = np.full(H, np.nan, np.float32); b[:hb] = v[n0 - hb:n0]; pre_act.append(b)
sc = np.array(sc); Qp = fc(post_ctx, H); Qb = fc(pre_ctx, H)
fp, fb = feats(Qp, np.stack(post_act), sc), feats(Qb, np.stack(pre_act), sc)
D = pd.DataFrame(index=pd.Index(ids, name="id"))
for k in fp: D[f"tx_post_{k}"] = fp[k]; D[f"tx_pre_{k}"] = fb[k]; D[f"tx_diff_{k}"] = fp[k] - fb[k]
D.to_parquet(OUT); print(D.shape, f"{time.time()-t0:.0f}s")
if NS != "all":
    from sklearn.metrics import roc_auc_score
    y = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"].reindex(D.index)
    au = {c: max(a := roc_auc_score(y, D[c].fillna(D[c].median())), 1 - a) for c in D}; print(sorted(au.items(), key=lambda kv: -kv[1])[:10])
