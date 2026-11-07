"""Persistence of the change after the boundary (probe/vol_persist.py: real breaks hold to the end, decoys and natural
wander fade). The part after n0 is cut into W equal windows; per window the change against the pre part is measured:
  vol  log sd of AR(5) residuals (whole-series fit) in the window - log sd of the pre residuals
  lvl  (mean x in the window - mean x pre) / sd x pre
Signed by the direction of the whole-post change (s = sign of the post-wide value), per kind:
  pz_<k>_min   min_w s*w   (does EVERY window keep the change?)     pz_<k>_last  s*w of the last window
  pz_<k>_first s*w of the first window                              pz_<k>_slope slope of s*w over the windows
  pz_<k>_keep  min / max of s*w (share that is left in the weakest window)
plus a sliding view over the whole post part: pz_<k>_frac = share of rolling windows (length R, step R/2) whose s*w
exceeds half the post-wide change.

    python probe/persist_feats.py     # train + test -> outputs/factory/persist.parquet
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

W, R, P = 4, 40, 5


def one(x, n0):
    n = len(x)
    A = np.column_stack([np.ones(n - P)] + [x[P - k:n - k] for k in range(1, P + 1)])
    e = np.r_[np.zeros(P), x[P:] - A @ np.linalg.lstsq(A, x[P:], rcond=None)[0]]
    pre_e, pre_x = e[P:n0], x[:n0]
    sde, mx, sdx = pre_e.std() + 1e-12, pre_x.mean(), pre_x.std() + 1e-12
    kinds = {"vol": lambda seg_e, seg_x: np.log(seg_e.std() / sde + 1e-12),
             "lvl": lambda seg_e, seg_x: (seg_x.mean() - mx) / sdx}
    post_e, post_x = e[n0:], x[n0:]
    cut = np.linspace(0, len(post_e), W + 1).astype(int)
    starts = np.arange(0, max(len(post_e) - R, 0) + 1, R // 2)
    out = {}
    for k, f in kinds.items():
        tot = f(post_e, post_x)
        s = 1.0 if tot >= 0 else -1.0
        w = np.array([s * f(post_e[a:b], post_x[a:b]) for a, b in zip(cut[:-1], cut[1:])])
        roll = np.array([s * f(post_e[a:a + R], post_x[a:a + R]) for a in starts])
        out.update({f"pz_{k}_min": w.min(), f"pz_{k}_last": w[-1], f"pz_{k}_first": w[0],
                    f"pz_{k}_slope": np.polyfit(np.arange(W), w, 1)[0], f"pz_{k}_keep": w.min() / (w.max() + 1e-9),
                    f"pz_{k}_frac": float((roll > abs(tot) / 2).mean())})
    return out


if __name__ == "__main__":
    t0 = time.time()
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in ("train", "test")]).sort_values(["id", "time"])
    items = [(str(k), g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    rows = Parallel(n_jobs=30, batch_size=64)(delayed(one)(v, n0) for _, v, n0 in items)
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    G.to_parquet("outputs/factory/persist.parquet")
    print(f"{len(G)} series in {time.time() - t0:.1f}s")
