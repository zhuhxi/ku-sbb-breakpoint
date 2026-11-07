"""Clean re-write of the guessed KU-SBB generator (replaces the patch history in probe/calib_break.py).
Every number that is not pinned down by the data is a key of DEFAULT, so a search can tune it.
Pinned by the data (LESSONS 2026-11-04/05): lengths, boundary range, outer shell, AR series at unit size, burn-in,
one recursion across the boundary, decoys only in no-break series, breaks last to the end, full float64.

    from ku_gen import gen, DEFAULT
    x, n0, info = gen(DEFAULT, seed, brk=True)
"""
import numpy as np
from scipy.signal import lfilter

DEFAULT = dict(
    p_struct=[.42, .17, .10, .15, .09, .07],          # WN AR1 AR2 SIN RW LL
    p_bimodal=.05, p_t=.50, t_df=(4.5, 10.0), garch_p=.25,
    ar1=(-.5, .7), ar2_p1=.9, ar2_p2=.5,
    sin_amp=(.2, 2.5), sin_scale=.86,
    ll_sd=(.02, .1), trend_p=.08, trend=(.5, 2.5),
    p_kind=[.40, .40, .10, .10],                       # spread level dep shape
    spread_small=(.05, .35), spread_ln2=.17,           # |log k|: U(small) or ln 2
    lvl_scale=1.25, lvl_weak=(.01, .49), lvl_mid=.11, lvl_big=.13, lvl_mid_v=(.8, .9), lvl_big_v=(1.8, 2.2),
    lvl_delay=.10,
    dep_small=(.05, .2), dep_big=.10, dep_big_v=.5,
    decoy_p=.10, decoy_size=(.4, 1.6), decoy_len=(10, 22),
    vburst_p=.06, vburst_k=(1.5, 2.8), vburst_len=(8, 18),
    drift_p=.08, drift_size=(.5, 2.0),
    dep_renorm=1.0,                                    # struct_tests S3: variance kept after the AR coefficient changes
    lvl_icp=.6,                                        # struct_tests S1: share of level breaks through the AR intercept
)
BURN = 200
_CAT = {}
_LAST = {}                                            # noise() records what it drew (for the hidden-parameter targets)


def catalog():
    """Per-series fits of the real no-break training series (genA): noise type / df, AR1 phi, sine amplitude."""
    if not _CAT:
        import pandas as pd
        A = pd.read_parquet("outputs/factory/genA.parquet")
        A.index = A.index.astype(str)
        L = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
        A = A.loc[L.index[L == 0]]
        _CAT["noise"] = A[["g_noise", "g_df"]].to_numpy()
        _CAT["phi"] = A.loc[A.g_struct == 1, "g_phi1"].to_numpy()
        _CAT["sin"] = A.loc[A.g_struct.isin([6, 7]), "g_sinamp"].to_numpy()
    return _CAT


def noise(r, n, P):
    if P.get("cat_noise"):                              # bootstrap (type, df) from the real catalog
        ty, df = catalog()["noise"][r.integers(len(catalog()["noise"]))]
        u = 0.0 if ty == 2 else (P["p_bimodal"] + 1e-9 if ty == 1 else 1.0)
        z = None
        if ty == 1:
            df = max(df, 2.2)
            z = r.standard_t(df, n) / np.sqrt(df / (df - 2))
        elif ty == 0:
            z = r.standard_normal(n)
        else:
            e = r.choice([-1, 1], n) * r.uniform(1.0, 2.5) + r.standard_normal(n)
            return e / e.std()
        return garchify(r, z, P)
    u = r.random()
    _LAST.update(ntype=0, ndf=0.0, garch=0)
    if u < P["p_bimodal"]:
        _LAST["ntype"] = 2
        e = r.choice([-1, 1], n) * r.uniform(1.0, 2.5) + r.standard_normal(n)
        return e / e.std()
    if u < P["p_bimodal"] + (1 - P["p_bimodal"]) * P["p_t"]:
        df = r.uniform(*P["t_df"]); _LAST.update(ntype=1, ndf=df)
        z = r.standard_t(df, n) / np.sqrt(df / (df - 2))
    else:
        z = r.standard_normal(n)
    return garchify(r, z, P)


def garchify(r, z, P):
    n = len(z)
    if r.random() < P["garch_p"]:
        _LAST["garch"] = 1
        a, b = r.uniform(.05, .15), r.uniform(*P.get("garch_b", (.5, .8)))
        b = min(b, .97 - a)
        e, s2 = np.empty(n), 1.0
        for t in range(n):
            if t:
                s2 = (1 - a - b) + a * e[t - 1] ** 2 + b * s2
            e[t] = np.sqrt(s2) * z[t]
        return e
    return z


def ar2_var(p1, p2):
    return (1 - p2) / ((1 + p2) * ((1 - p2) ** 2 - p1 ** 2))


def kurtosis_(v):
    v = v - v.mean()
    return (v ** 4).mean() / (v ** 2).mean() ** 2 - 3


def sgn(r):
    return r.choice([-1.0, 1.0])


def csize(r, P, scale, sk=None):
    """Continuous break size (size_mode 'hn' half-normal / 'exp' exponential): many tiny, fewer large, no bins."""
    if P["size_mode"] == "ln":   # lognormal: median = scale, long right tail, no cap
        return float(np.exp(np.log(scale) + P.get(sk, P.get("ln_sig", 1.0)) * r.standard_normal()))
    return abs(r.normal(0, scale)) if P["size_mode"] == "hn" else r.exponential(scale)


def gen(P, seed, brk):
    r = np.random.default_rng(seed)
    n = int(r.integers(256, 1025))
    n0 = int(r.uniform(.3, .7) * n)
    BU = P.get("burn", BURN)
    N = n + BU
    b0 = n0 + BU                                       # boundary inside the burned-in array
    t = np.arange(n)
    fam = ["WN", "AR1", "AR2", "SIN", "RW", "LL"][r.choice(6, p=np.asarray(P["p_struct"]) / np.sum(P["p_struct"]))]
    e = noise(r, N, P)
    nz = dict(_LAST)
    kind = r.choice(["spread", "level", "dep", "shape"], p=np.asarray(P["p_kind"]) / np.sum(P["p_kind"])) if brk else None
    kind2 = None
    if brk and r.random() < P.get("couple_p", 0):       # a second, different change at the same boundary
        kind2 = r.choice([k for k in ("spread", "level", "dep", "shape") if k != kind])
    kinds = {kind, kind2}
    hid = dict(ntype=nz.get("ntype", 0), inv_df=1 / nz["ndf"] if nz.get("ndf") else 0.0, garch=nz.get("garch", 0), sin_amp=0.0, sin_per=0, p1=0.0, p2=0.0, dep_d=0.0, log_k=0.0, lvl_d=0.0, lvl_mode=0, shape=0, decoy=0, vburst=0, drift=0, trend=0)
    if brk and P.get("kind_mode") == "switch":         # every change switched on independently (at least the main one)
        kinds |= {k for k, q in zip(("spread", "level", "dep", "shape"), P["p_kind"]) if r.random() < P.get("switch_k", .3) * q}
    if "shape" in kinds:
        hid["shape"] = 1
        sm = P.get("shape_mode", "redraw")
        if sm == "heavy":                                # tails get heavier after the boundary
            df = r.uniform(3, 5); e[b0:] = r.standard_t(df, N - b0) / np.sqrt(df / (df - 2))
        elif sm == "flip":                               # Gaussian <-> t: the family is forced to change
            if kurtosis_(e[:b0]) > .5:
                e[b0:] = r.standard_normal(N - b0)
            else:
                df = r.uniform(3, 6); e[b0:] = r.standard_t(df, N - b0) / np.sqrt(df / (df - 2))
        else:
            e[b0:] = noise(r, N - b0, P)
    p1 = p2 = 0.0
    sin_ar = fam == "SIN" and P.get("sin_ar_p", 0) > 0 and r.random() < P["sin_ar_p"]
    if sin_ar:
        p1 = r.uniform(.2, .7)
    if fam in ("AR1", "AR2"):
        if fam == "AR1":
            p1 = float(np.clip(r.choice(catalog()["phi"]), -.95, .95)) if P.get("cat_phi") else r.uniform(*P["ar1"])
        elif fam == "AR2":
            while True:
                p1, p2 = r.uniform(-P["ar2_p1"], P["ar2_p1"]), r.uniform(-P["ar2_p2"], P["ar2_p2"])
                if p2 + p1 < 1 and p2 - p1 < 1:
                    break
    if "dep" in kinds:                                   # AR(1) coefficient changes at the boundary, one recursion
        d = csize(r, P, P["dep_cs"], "ln_sig_dep") if P.get("size_mode") else P["dep_big_v"] if r.random() < P["dep_big"] else r.uniform(*P["dep_small"])
        q1 = float(np.clip(p1 + sgn(r) * d, -.95, .95))
        hid["dep_d"] = q1 - p1
        if fam == "AR2":                                # keep the changed AR(2) stationary
            q1 = float(np.clip(q1, p2 - 1 + .02, 1 - p2 - .02))
        z = np.empty(N)
        z[:b0] = lfilter([1], [1, -p1, -p2], e[:b0])
        z[b0:] = lfilter([1], [1, -q1, -p2], e[b0:], zi=np.array([q1 * z[b0 - 1] + p2 * z[b0 - 2], p2 * z[b0 - 1]]))[0]
    else:
        z = lfilter([1], [1, -p1, -p2], e)
    hid["p1"], hid["p2"] = p1, p2
    if fam in ("AR1", "AR2") or sin_ar:
        z = z / np.sqrt(ar2_var(p1, p2))                # real AR series have unit marginal size
    if "dep" in kinds and r.random() < P.get("dep_renorm", 0):   # keep the marginal size after the coefficient change
        z[b0:] *= np.sqrt(ar2_var(p1, p2) / ar2_var(q1, p2)) if fam in ("AR1", "AR2") else 1 / np.sqrt(ar2_var(q1, p2) / ar2_var(p1, p2))
    z = z[BU:]
    e_sd = e[BU:b0].std()
    if "spread" in kinds:
        k = csize(r, P, P["spread_cs"], "ln_sig_spr") if P.get("size_mode") else np.log(2) if r.random() < P["spread_ln2"] else r.uniform(*P["spread_small"])
        kout = np.exp(sgn(r) * k)
        hid["log_k"] = float(np.log(kout))
        if P.get("spread_target", "noise") == "noise":
            z[n0:] *= kout                              # on the output of the noise process
        elif P["spread_target"] == "innov" and "dep" not in kinds:   # on the innovation sigma: AR dynamics carry it in
            e2 = e.copy(); e2[b0:] *= kout
            zz = lfilter([1], [1, -p1, -p2], e2)
            if fam in ("AR1", "AR2") or sin_ar:
                zz = zz / np.sqrt(ar2_var(p1, p2))
            z = zz[BU:]
    if fam == "SIN":
        amp = r.choice(catalog()["sin"]) * z.std() if P.get("cat_sin") else r.uniform(*P["sin_amp"])
        per = r.integers(12, 33); hid["sin_amp"], hid["sin_per"] = float(amp), int(per)
        x = (z + amp * np.sin(2 * np.pi * t / per + r.uniform(0, 6.3))) * P["sin_scale"]
    elif fam == "RW":
        x = np.cumsum(z)
    elif fam == "LL":
        if P.get("ll_mode") == "arma":                  # ARMA(1,1) near a unit root instead of level walk + noise
            ph, th = r.uniform(.85, .97), r.uniform(-.6, -.3)
            x = lfilter([1, th], [1, -ph], z)
            x = x / x.std()
        else:
            x = z + np.cumsum(noise(r, n, P) * r.uniform(*P["ll_sd"]))
    else:
        x = z
    if "spread" in kinds and P.get("spread_target", "noise") == "output":   # on the whole series (sine / walk included)
        mx = x[:n0].mean()
        x = x.copy(); x[n0:] = mx + (x[n0:] - mx) * kout
    if "level" in kinds:
        u = r.random()
        if P.get("size_mode"):
            d = csize(r, P, P["lvl_cs"], "ln_sig_lvl")
        elif u < P["lvl_big"]:
            d = r.uniform(*P["lvl_big_v"])
        elif u < P["lvl_big"] + P["lvl_mid"]:
            d = r.uniform(*P["lvl_mid_v"])
        else:
            d = r.uniform(*P["lvl_weak"])
        d *= sgn(r) * P["lvl_scale"] * (x[:n0].std() if P.get("lvl_unit") == "series" else e_sd)
        hid["lvl_d"] = float(d / (x[:n0].std() if P.get("lvl_unit") == "series" else e_sd))
        step = np.zeros(n)
        if fam in ("AR1", "AR2") and r.random() < P.get("lvl_icp", 0):
            hid["lvl_mode"] = 1   # through the intercept: AR dynamics carry the jump in
            st_ = np.zeros(n); st_[n0:] = d * (1 - p1 - p2)
            step = lfilter([1], [1, -p1, -p2], st_)
        elif r.random() < P["lvl_delay"]:
            lag = int(r.integers(4, 21)); hid["lvl_mode"] = 2
            if r.random() < .5:
                step[n0 + lag:] = d                     # late
            else:
                step[n0:n0 + lag] = d * np.arange(1, lag + 1) / lag   # short transition
                step[n0 + lag:] = d
        else:
            step[n0:] = d
        x = x + step
    if (fam in ("WN", "AR1", "AR2") or P.get("trend_fams") == "all") and r.random() < P["trend_p"]:
        x = x + sgn(r) * r.uniform(*P["trend"]) * x.std() * t / n
        hid["trend"] = 1
    NB = P.get("nuis_brk", ("decoy", "vburst", "drift") if P.get("decoy_in_breaks") else ())
    if not brk or NB:
        sd = x[:n0].std()
        if (not brk or "decoy" in NB) and r.random() < P["decoy_p"]:
            L = int(r.integers(*P["decoy_len"]))
            s0 = n0 if P.get("decoy_where", "start") == "start" else int(r.integers(n0, max(n - L, n0 + 1)))
            w = np.sin(np.pi * (np.arange(L) + .5) / L)[:n - s0]; hid["decoy"] = 1
            x[s0:s0 + L] += sgn(r) * r.uniform(*P["decoy_size"]) * sd * w
        if (not brk or "vburst" in NB) and r.random() < P["vburst_p"]:
            L = int(r.integers(*P["vburst_len"]))
            hid["vburst"] = 1
            env = 1 + (r.uniform(*P["vburst_k"]) - 1) * np.sin(np.pi * (np.arange(L) + .5) / L)
            mu = x[:n0].mean()
            seg = x[n0:n0 + L]
            x[n0:n0 + L] = mu + (seg - mu) * env[:len(seg)]
        if (not brk or "drift" in NB) and r.random() < P["drift_p"]:
            hid["drift"] = 1
            x[n0:] += sgn(r) * r.uniform(*P["drift_size"]) * e_sd * (t[n0:] - n0) / (n - n0)
    if brk and P.get("shock_p", 0) and r.random() < P["shock_p"]:        # transient post-break shock: extra noise decaying from the boundary
        L = n - n0; j = np.arange(L)
        x[n0:] += r.uniform(*P.get("shock_a", (.5, 1.5))) * e_sd * np.exp(-j / r.uniform(*P.get("shock_tau", (3, 15)))) * r.standard_normal(L)
    if r.random() < P.get("spike_p", 0):                # isolated shocks anywhere (spike bank idea)
        sdn = np.std(np.diff(x)) / np.sqrt(2)
        for j in r.integers(1, n - 1, int(r.integers(1, 4))):
            x[j] += sgn(r) * r.uniform(*P.get("spike_size", (5, 10))) * sdn
    if not brk and fam in ("WN", "AR1", "AR2") and r.random() < P.get("excursion_p", 0):   # hard negative: slow wander
        x = x + np.cumsum(r.standard_normal(n)) * r.uniform(*P.get("excursion_sd", (.03, .08))) * x.std()
    x = r.uniform(-2, 2) + 2 ** r.uniform(-1, 1) * x
    return x, n0, dict(family=fam, kind=kind, kind2=kind2, **hid)
