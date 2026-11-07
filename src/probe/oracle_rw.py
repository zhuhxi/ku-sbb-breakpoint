"""Oracle ceiling for RANDOM-WALK series (same simplified best2 world as probe/oracle_ceiling.py: no GARCH / bimodal / decoys).
x = shell(cumsum(z)); z = unit-variance Gaussian / t noise; spread, dep (AR(1) on the increments), shape (noise re-drawn) act on
z; level is a step on x = an impulse (or a 4-20 step ramp / late jump) on the increments. The oracle scores the increments with
the true noise family / df / shell scale (break averaged over its prior); the plug-in version estimates the scale and the noise
family from the pre increments. Practical comparison: LightGBM on the 119 fast features, same fakes.
    python probe/oracle_rw.py N"""
import sys, time
import lightgbm as lgb, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "probe")
import oracle_ceiling as O, oracle_plugin as OP
P, PK, M = O.P, O.PK, O.M


def lvl_profile(r, n, n0, esd):
    """level break on x as increments: impulse / ramp / late jump (no intercept form for RW)."""
    u = r.random()
    d = r.uniform(*P["lvl_big_v"]) if u < P["lvl_big"] else r.uniform(*P["lvl_mid_v"]) if u < P["lvl_big"] + P["lvl_mid"] else r.uniform(*P["lvl_weak"])
    size = d; d *= r.choice([-1., 1.]) * P["lvl_scale"] * esd
    step = np.zeros(n)
    if r.random() < P["lvl_delay"]:
        lag = int(r.integers(4, 21))
        if r.random() < .5: step[n0 + lag:] = d
        else: step[n0:n0 + lag] = d * np.arange(1, lag + 1) / lag; step[n0 + lag:] = d
    else:
        step[n0:] = d
    return np.diff(step), size


def gen(seed, brk):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025)); n0 = int(r.uniform(.3, .7) * n)
    ty, df, z = O.tdraw(r, n)
    kind = r.choice(["spread", "level", "dep", "shape"], p=PK) if brk else "none"; size = 0.0
    if kind == "shape": ty2, df2, z[n0:] = O.tdraw(r, n - n0); size = float(ty2 != ty)
    if kind == "dep":
        q1, size = O.draw_dep(r, 0.0)
        f = z.copy(); f[n0:] = O.lfilter([1], [1, -q1], z[n0:], zi=[q1 * z[n0 - 1]])[0]; z = np.r_[z[:n0], f[n0:] / O.sd_ar(q1)]
    esd = z[:n0].std()
    if kind == "spread": k = O.draw_spread(r); z[n0:] *= np.exp(k); size = abs(k)
    x = np.cumsum(z)
    if kind == "level": dstep, size = lvl_profile(r, n, n0, esd); x = x + np.r_[0, np.cumsum(dstep)]
    a, b = r.uniform(-2, 2), 2 ** r.uniform(-1, 1)
    return a + b * x, n0, dict(b=b, ty=ty, df=df, esd=esd, kind=kind, size=size)


def score(x, n0, b, ty, df, esd, seed):
    r = np.random.default_rng(seed + 10 ** 9)
    z = np.diff(x) / b; m0 = n0 - 1; n = len(x)   # increments; z[t-1] = x[t] - x[t-1]
    l0 = O.ll_ar(np.r_[0, z], n0, 0.0, 0.0, 1.0, ty, df)
    L = [np.log(PK[0]) + logsumexp([O.ll_ar(np.r_[0, z], n0, 0.0, 0.0, 1 / np.exp(O.draw_spread(r)), ty, df) for _ in range(M)]) - np.log(M)]
    L.append(np.log(PK[1]) + logsumexp([O.ll_ar(np.r_[0, z - lvl_profile(r, n, n0, esd)[0]], n0, 0.0, 0.0, 1.0, ty, df) for _ in range(M)]) - np.log(M))
    dl = []
    for _ in range(M):
        q1, _d = O.draw_dep(r, 0.0); dl.append(O.ll_ar(np.r_[0, z], n0, 0.0, q1, O.sd_ar(q1), ty, df))
    L.append(np.log(PK[2]) + logsumexp(dl) - np.log(M))
    sl = [np.log(1 - P["p_t"]) + O.ll_ar(np.r_[0, z], n0, 0.0, 0.0, 1.0, ty, df, "g", 0.0)]
    sl.append(np.log(P["p_t"]) + logsumexp([O.ll_ar(np.r_[0, z], n0, 0.0, 0.0, 1.0, ty, df, "t", d2) for d2 in np.linspace(*P["t_df"], 16)]) - np.log(16))
    L.append(np.log(PK[3]) + logsumexp(sl))
    return float(logsumexp(L) - l0)


def one(s):
    x, n0, q = gen(s, bool(s % 2))
    true = score(x, n0, q["b"], q["ty"], q["df"], q["esd"], s)
    d = np.diff(x[:n0]); est = OP.estimate(np.cumsum(np.r_[0, d]) if False else d, len(d))   # noise family on pre increments
    plug = score(x, n0, d.std(), est["ty"], est["df"], 1.0, s)
    return dict(oracle=true, plug=plug, label=s % 2, kind=q["kind"])


if __name__ == "__main__":
    import oracle_rw as RW
    t0 = time.time(); NS = int(sys.argv[1])
    D = pd.DataFrame(Parallel(n_jobs=30, batch_size=16)(delayed(RW.one)(s) for s in range(NS)))
    print(f"RW simplified world, {NS} fakes: oracle (true params) {roc_auc_score(D.label, D.oracle):.4f} | plug-in (estimated) {roc_auc_score(D.label, D.plug):.4f}  {time.time() - t0:.0f}s")
    neg = D[D.label == 0]
    for k in ("spread", "level", "dep", "shape"):
        g = pd.concat([neg, D[D.kind == k]]); print(f"  {k:6s} oracle {roc_auc_score(g.label, g.oracle):.4f}  plug-in {roc_auc_score(g.label, g.plug):.4f}")
    D.to_parquet("outputs/factory/oracle_rw.parquet")
