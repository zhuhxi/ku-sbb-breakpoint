"""Oracle AUC ceiling under the guessed generator (best2 priors), WN + AR1 series.
Simplified world: no GARCH / bimodal noise, no trend, no decoy / vburst / drift, one change kind.
The oracle KNOWS the structure, AR phi, noise family + df, the outer shell and the pre noise sd, but NOT whether / how the
series broke: score = log p(x | break) - log p(x | no break), the break averaged over its prior (kind, size, mode)
by Monte Carlo. No method can beat it on this data, so its AUC is an upper bound.

    python probe/oracle_ceiling.py NSERIES   -> outputs/factory/oracle.parquet (score, label, family, kind, size)
"""
import json, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter
from scipy.special import gammaln, logsumexp
from sklearn.metrics import roc_auc_score

P = json.load(open("outputs/factory/ku_gen_best2.json"))
PK = np.array(P["p_kind"]) / sum(P["p_kind"])          # spread level dep shape
BU, M = 200, 64


def tdraw(r, n):
    if r.random() < P["p_t"]:
        df = r.uniform(*P["t_df"]); return "t", df, r.standard_t(df, n) / np.sqrt(df / (df - 2))
    return "g", 0.0, r.standard_normal(n)


def logf(e, ty, df):
    if ty == "g":
        return -.5 * e ** 2 - .5 * np.log(2 * np.pi)
    s = np.sqrt((df - 2) / df)
    return gammaln((df + 1) / 2) - gammaln(df / 2) - .5 * np.log(df * np.pi) - np.log(s) - (df + 1) / 2 * np.log1p((e / s) ** 2 / df)


def sd_ar(p): return 1 / np.sqrt(1 - p ** 2)


def draw_level(r, n, n0, fam, p1, esd):
    u = r.random()
    d = r.uniform(*P["lvl_big_v"]) if u < P["lvl_big"] else r.uniform(*P["lvl_mid_v"]) if u < P["lvl_big"] + P["lvl_mid"] else r.uniform(*P["lvl_weak"])
    size = d; d *= r.choice([-1., 1.]) * P["lvl_scale"] * esd
    step = np.zeros(n)
    if fam == "AR1" and r.random() < P["lvl_icp"]:
        st = np.zeros(n); st[n0:] = d * (1 - p1); step = lfilter([1], [1, -p1], st)
    elif r.random() < P["lvl_delay"]:
        lag = int(r.integers(4, 21))
        if r.random() < .5: step[n0 + lag:] = d
        else: step[n0:n0 + lag] = d * np.arange(1, lag + 1) / lag; step[n0 + lag:] = d
    else:
        step[n0:] = d
    return step, size


def draw_spread(r):
    k = np.log(2) if r.random() < P["spread_ln2"] else r.uniform(*P["spread_small"])
    return r.choice([-1., 1.]) * k


def draw_dep(r, p1):
    d = P["dep_big_v"] if r.random() < P["dep_big"] else r.uniform(*P["dep_small"])
    return float(np.clip(p1 + r.choice([-1., 1.]) * d, -.95, .95)), d


def gen(seed, brk):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025)); n0 = int(r.uniform(.3, .7) * n); N, b0 = n + BU, n0 + BU
    fam = "WN" if r.random() < .42 / .59 else "AR1"
    p1 = 0.0 if fam == "WN" else r.uniform(*P["ar1"])
    ty, df, e = tdraw(r, N)
    kind = r.choice(["spread", "level", "dep", "shape"], p=PK) if brk else "none"; size = 0.0
    if kind == "shape":
        ty2, df2, e[b0:] = tdraw(r, N - b0); size = float(ty2 != ty)
    if kind == "dep":
        q1, size = draw_dep(r, p1)
        f = np.empty(N); f[:b0] = lfilter([1], [1, -p1], e[:b0])
        f[b0:] = lfilter([1], [1, -q1], e[b0:], zi=[q1 * f[b0 - 1]])[0]
        z = np.r_[f[:b0] / sd_ar(p1), f[b0:] / sd_ar(q1)]
    else:
        z = lfilter([1], [1, -p1], e) / sd_ar(p1)
    z = z[BU:]; esd = e[BU:b0].std()
    if kind == "spread":
        k = draw_spread(r); z[n0:] *= np.exp(k); size = abs(k)
    if kind == "level":
        step, size = draw_level(r, n, n0, fam, p1, esd); z = z + step
    a, b = r.uniform(-2, 2), 2 ** r.uniform(-1, 1)
    return a + b * z, n0, dict(fam=fam, p1=p1, ty=ty, df=df, a=a, b=b, esd=esd, kind=kind, size=size)


def ll_ar(z, n0, p1, c_post, sc_post, ty, df, ty2=None, df2=None):
    """log density of z (t >= 1): filt = z * sd(p1) before, z * sc_post after; AR coef p1 before, c_post after."""
    f = np.r_[z[:n0] * sd_ar(p1), z[n0:] * sc_post]
    e = f[1:] - np.r_[np.full(n0 - 1, p1), np.full(len(z) - n0, c_post)] * f[:-1]
    lj = (n0 - 1) * np.log(sd_ar(p1)) + (len(z) - n0) * np.log(sc_post)
    if ty2 is None:
        return logf(e, ty, df).sum() + lj
    return logf(e[:n0 - 1], ty, df).sum() + logf(e[n0 - 1:], ty2, df2).sum() + lj


def score(x, n0, q, seed, parts=False):
    r = np.random.default_rng(seed + 10 ** 9)
    z = (x - q["a"]) / q["b"]; p1, ty, df = q["p1"], q["ty"], q["df"]; n = len(z); s1 = sd_ar(p1)
    l0 = ll_ar(z, n0, p1, p1, s1, ty, df)
    L = []
    L.append(np.log(PK[0]) + logsumexp([ll_ar(z, n0, p1, p1, s1 / np.exp(draw_spread(r)), ty, df) for _ in range(M)]) - np.log(M))
    L.append(np.log(PK[1]) + logsumexp([ll_ar(z - draw_level(r, n, n0, q["fam"], p1, q["esd"])[0], n0, p1, p1, s1, ty, df) for _ in range(M)]) - np.log(M))
    dl = []
    for _ in range(M):
        q1, _d = draw_dep(r, p1); dl.append(ll_ar(z, n0, p1, q1, sd_ar(q1), ty, df))
    L.append(np.log(PK[2]) + logsumexp(dl) - np.log(M))
    sl = [np.log(1 - P["p_t"]) + ll_ar(z, n0, p1, p1, s1, ty, df, "g", 0.0)]
    dfs = np.linspace(*P["t_df"], 16)
    sl.append(np.log(P["p_t"]) + logsumexp([ll_ar(z, n0, p1, p1, s1, ty, df, "t", d2) for d2 in dfs]) - np.log(16))
    L.append(np.log(PK[3]) + logsumexp(sl))
    if parts:
        return dict(lr_all=float(logsumexp(L) - l0), **{f'lr_{k}': float(v - l0) for k, v in zip(('spread', 'level', 'dep', 'shape'), L)})
    return float(logsumexp(L) - l0)


def one(s):
    x, n0, q = gen(s, bool(s % 2))
    return dict(score=score(x, n0, q, s), label=s % 2, fam=q["fam"], kind=q["kind"], size=q["size"], n=len(x))


if __name__ == "__main__":
    sys.path.insert(0, "probe"); import oracle_ceiling as O
    t0 = time.time(); NS = int(sys.argv[1])
    D = pd.DataFrame(Parallel(n_jobs=30, batch_size=16)(delayed(O.one)(s) for s in range(NS)))
    D.to_parquet("outputs/factory/oracle.parquet")
    neg = D[D.label == 0]
    print(f"oracle AUC (WN+AR1, simplified world, {NS} series) {roc_auc_score(D.label, D.score):.4f}  {time.time() - t0:.0f}s")
    for f in ("WN", "AR1"):
        g = D[D.fam == f]; print(f"  {f:4s} {roc_auc_score(g.label, g.score):.4f}")
    for k in ("spread", "level", "dep", "shape"):
        g = pd.concat([neg, D[D.kind == k]]); print(f"  kind {k:6s} (vs all negatives) {roc_auc_score(g.label, g.score):.4f}   share of breaks {np.mean(D[D.label == 1].kind == k):.2f}")
