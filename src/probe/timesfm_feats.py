"""TimesFM-2.5 (public 200M weights, local GPU) forecast-error features, same design as probe/chronos_feats.py: forecast the first H
post points from the pre context and a back-test on the last H pre points; prefix tfm_. REVSEQ=1: time-reversed series.
    python probe/timesfm_feats.py OUT"""
import os, sys, time, numpy as np, pandas as pd, timesfm
OUT = sys.argv[1]; H, CTX = int(os.environ.get("HTF", 64)), 512; PFX = os.environ.get("PFX", "tfm"); t0 = time.time()
M = timesfm.TimesFM_2p5_200M_torch.from_pretrained(timesfm.TimesFM_2p5_200M_torch.DEFAULT_REPO_ID)
M.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=max(128, H), normalize_inputs=True, per_core_batch_size=512, fix_quantile_crossing=True))
S = pd.concat([pd.read_parquet("data/train_series.parquet"), pd.read_parquet("data/test_series.parquet")]); S["id"] = S["id"].astype(str); S = S.sort_values(["id", "time"], kind="stable")
if os.environ.get("REVSEQ"):
    n = S.groupby("id")["time"].transform("max"); S = S.assign(time=n - S["time"], period=1 - S["period"]).sort_values(["id", "time"], kind="stable")
ids = list(pd.read_csv("data/train_labels.csv", dtype={"id": str}).id) + list(pd.read_csv("data/sample_submission.csv", dtype={"id": str}).id)
G = {k: (g["value"].to_numpy(np.float64), int((g["period"].to_numpy() == 0).sum())) for k, g in S.groupby("id", sort=False)}
def fc(ctxs):
    Q = []
    for i in range(0, len(ctxs), 4096):
        _, q = M.forecast(horizon=H, inputs=ctxs[i:i + 4096]); Q.append(np.asarray(q)[:, :H, [1, 5, 9]])
    return np.concatenate(Q)
def feats(Q, A, sc):
    lo, md, hi = Q[..., 0], Q[..., 1], Q[..., 2]; w = (hi - lo) + 1e-6 * sc[:, None]; d = {}
    for m in ((8, 16, 32, 64) if H <= 64 else (32, 64, 96, 128)):
        e = A[:, :m] - md[:, :m]; d[f"e{m}"] = np.nanmean(np.abs(e) / w[:, :m], 1); d[f"b{m}"] = np.nanmean(e, 1) / (np.nanmean(w[:, :m], 1) + 1e-9)
        d[f"out{m}"] = np.nanmean((A[:, :m] < lo[:, :m]) | (A[:, :m] > hi[:, :m]), 1); d[f"sd{m}"] = np.log((np.nanstd(A[:, :m], 1) + 1e-6) / (np.nanmean(w[:, :m], 1) + 1e-6))
    return d
pc, pa, bc, ba, sc = [], [], [], [], []
for k in ids:
    v, n0 = G[k]; sc.append(v[:n0].std() + 1e-6)
    pc.append(v[max(0, n0 - CTX):n0]); a = np.full(H, np.nan); seg = v[n0:n0 + H]; a[:len(seg)] = seg; pa.append(a)
    hb = min(H, n0 // 3); bc.append(v[max(0, n0 - hb - CTX):n0 - hb]); b = np.full(H, np.nan); b[:hb] = v[n0 - hb:n0]; ba.append(b)
sc = np.array(sc); Qp = fc(pc); Ap = np.stack(pa); fp = feats(Qp, Ap, sc); fb = feats(fc(bc), np.stack(ba), sc)
if os.environ.get("PTZ"):   # per-step standardised errors of the first 16 post points (prefix tfz_) instead of summaries
    w = Qp[..., 2] - Qp[..., 0] + 1e-6 * sc[:, None]; Z = np.clip((Ap - Qp[..., 1]) / w, -20, 20)
    D = pd.DataFrame(Z[:, :16], index=pd.Index(ids, name="id"), columns=[f"tfz_{t}" for t in range(16)]); D.to_parquet(OUT); print(D.shape); sys.exit()
D = pd.DataFrame(index=pd.Index(ids, name="id"))
for k in fp: D[f"{PFX}_post_{k}"] = fp[k]; D[f"{PFX}_pre_{k}"] = fb[k]; D[f"{PFX}_diff_{k}"] = fp[k] - fb[k]
D.to_parquet(OUT); print(D.shape, f"{time.time()-t0:.0f}s")
