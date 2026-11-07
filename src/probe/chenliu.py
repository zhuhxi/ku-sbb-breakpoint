"""Chen-Liu (1993) intervention statistics, used to tell a lasting step (LS, label 1) from a one-point outlier (AO) or a
decaying temporary change (TC) -- the three cases of the official notebook -- in wandering series too.
  1. AR(P) fitted on the pre part only (break-free reference); pi(B) = 1 - phi_1 B - ... - phi_P B^P applied to the whole
     series -> residuals e; sigma = 1.4826 * MAD of the pre residuals (robust)
  2. for a candidate time tau and pattern xi (AO impulse, LS step, TC step decaying with delta = .7):
     x = pi(B) xi,  omega = <e, x> / <x, x>,  tau-stat = omega * sqrt(<x, x>) / sigma   (Chen & Liu eq. for each type)
  3. tau over the window n0 - 2 .. n0 + 20 (late / transition jumps) -> max |LS| and where; also at n0 itself
  4. the same LS statistic at 60 random tau elsewhere -> rank of the boundary value (is the boundary special?)
  5. increment rank: |x_n0 - x_{n0-1}| among all |increments| (random-walk view)
Features cl_*. Quick mode evaluates on the error cases only (outputs/error_cases.csv): AUC missed (1) vs false alarm (0).

    python probe/chenliu.py errors        # quick check on the 5581 error cases
    python probe/chenliu.py               # train + test -> outputs/factory/chenliu.parquet
"""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

P, DELTA, WIN = 5, .7, 20


def pattern(kind, n, tau):
    xi = np.zeros(n)
    if kind == "AO":
        xi[tau] = 1
    elif kind == "LS":
        xi[tau:] = 1
    else:
        xi[tau:] = DELTA ** np.arange(n - tau)
    return xi


def stat(e, phi, kind, tau, sig):
    n = len(e)
    xi = pattern(kind, n + P, tau + P)
    x = xi[P:] - sum(phi[j] * xi[P - 1 - j:n + P - 1 - j] for j in range(P))
    xx = x @ x
    return (e @ x) / xx * np.sqrt(xx) / sig


def one(v, n0, seed):
    n = len(v)
    pre = v[:n0]
    A = np.column_stack([np.ones(n0 - P)] + [pre[P - k:n0 - k] for k in range(1, P + 1)])
    b = np.linalg.lstsq(A, pre[P:], rcond=None)[0]
    c, phi = b[0], b[1:]
    e = v[P:] - c - sum(phi[j] * v[P - 1 - j:n - 1 - j] for j in range(P))
    pe = e[:n0 - P]
    sig = 1.4826 * np.median(np.abs(pe - np.median(pe))) + 1e-12
    e = e - np.median(pe)
    i0 = n0 - P
    taus = range(max(i0 - 2, 1), min(i0 + WIN + 1, len(e) - 2))
    LS = np.array([stat(e, phi, "LS", t, sig) for t in taus])
    k = int(np.abs(LS).argmax())
    tb = list(taus)[k]
    at = {kd: stat(e, phi, kd, i0, sig) for kd in ("AO", "LS", "TC")}
    best = {kd: stat(e, phi, kd, tb, sig) for kd in ("AO", "TC")}
    rng = np.random.default_rng(seed)
    others = [t for t in rng.integers(20, len(e) - 20, 80) if abs(t - i0) > WIN + 5][:60]
    LSo = np.abs([stat(e, phi, "LS", t, sig) for t in others]) if others else np.array([0.0])
    d = np.abs(np.diff(v))
    return {"cl_ls_at": abs(at["LS"]), "cl_ao_at": abs(at["AO"]), "cl_tc_at": abs(at["TC"]),
            "cl_ls_win": abs(LS[k]), "cl_ls_lag": tb - i0,
            "cl_ls_edge": abs(LS[k]) - max(abs(best["AO"]), abs(best["TC"])),
            "cl_ls_edge_at": abs(at["LS"]) - max(abs(at["AO"]), abs(at["TC"])),
            "cl_ls_rank": float((LSo < abs(LS[k])).mean()), "cl_ls_vs_max": abs(LS[k]) / (LSo.max() + 1e-9),
            "cl_inc_rank": float((d < d[n0 - 1]).mean())}


if __name__ == "__main__":
    t0 = time.time()
    quick = len(sys.argv) > 1 and sys.argv[1] == "errors"
    parts = ("train",) if quick else ("train", "test")
    S = pd.concat([pd.read_parquet(f"data/{p}_series.parquet") for p in parts])
    S["id"] = S["id"].astype(str)
    if quick:
        E = pd.read_csv("outputs/error_cases.csv", dtype={"id": str}).set_index("id")
        S = S[S.id.isin(E.index)]
    S = S.sort_values(["id", "time"])
    items = [(k, g["value"].to_numpy(np.float64), int((g["period"] == 0).sum())) for k, g in S.groupby("id", sort=False)]
    rows = Parallel(n_jobs=30, batch_size=16)(delayed(one)(v, n0, i) for i, (_, v, n0) in enumerate(items))
    G = pd.DataFrame(rows, index=[k for k, _, _ in items]).astype(float)
    print(f"{len(G)} series in {time.time() - t0:.1f}s")
    if quick:
        from sklearn.metrics import roc_auc_score
        G = G.join(E[["label", "structure"]])
        for grp, D in [("all errors", G), ("RW", G[G.structure == "RW"]), ("ARMA11", G[G.structure == "ARMA11"]),
                       ("WN", G[G.structure == "WN"])]:
            au = {c: roc_auc_score(D.label, D[c]) for c in G.columns if c.startswith("cl_")}
            top = sorted(au.items(), key=lambda kv: -abs(kv[1] - .5))[:5]
            print(f"{grp:10s} n={len(D):4d} (missed {int(D.label.sum())}): " + "  ".join(f"{c} {a:.3f}" for c, a in top))
    else:
        G.to_parquet("outputs/factory/chenliu.parquet")
