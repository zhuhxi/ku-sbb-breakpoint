"""Synthetic pretraining data for sequence models: N fakes from probe/ku_gen.py (params PF, seeds S0..S0+N-1, label = s % 2),
converted to the boundary-aligned tensor layout of probe/seq_prep.py (z by pre median/MAD, AR(10)-pre residual, mask; PRE=960).
    python probe/sim_tensor.py N PF OUT.npz S0"""
import json, sys, numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, "probe")
from ku_gen import DEFAULT, gen
N, PF, OUT, S0 = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]); PRE, POST, q = 960, 736, 10; import os; REVP = float(os.environ.get("REVP", 0))
b = {k: (tuple(v) if isinstance(DEFAULT.get(k), tuple) else v) for k, v in json.load(open(PF)).items()}; P = dict(DEFAULT, **b)
def one(s):
    v, n0, inf = gen(P, S0 + s, bool(s % 2)); v = np.asarray(v, float); out = np.zeros((3, PRE + POST), np.float16)
    if REVP and np.random.default_rng(S0 + s + 7).random() < REVP: v = v[::-1].copy(); n0 = len(v) - n0   # time-reversed fake (same label)
    KI = {None: 0, 'spread': 1, 'level': 2, 'dep': 3, 'shape': 4}; FA = {'WN': 0, 'AR1': 1, 'AR2': 2, 'SIN': 3, 'RW': 4, 'LL': 5}
    meta = np.array([KI.get(inf['kind'], 0), FA.get(inf['family'], 0), abs(inf['log_k']), abs(inf['lvl_d']), abs(inf['dep_d']), inf['ntype'], inf['garch'], inf['decoy'], inf['trend']], np.float32)
    if n0 > PRE or len(v) - n0 > POST or n0 < 2 * q: return out, -1, meta
    x0 = v[:n0]; med = np.median(x0); mad = 1.4826 * np.median(np.abs(x0 - med)) + 1e-8; z = (v - med) / mad
    Zl = np.column_stack([np.ones(len(z) - q)] + [z[q - j - 1:len(z) - j - 1] for j in range(q)])
    c = np.linalg.lstsq(Zl[:n0 - q], z[q:n0], rcond=None)[0]; e = np.r_[np.zeros(q), z[q:] - Zl @ c]; e /= e[q:n0].std() + 1e-8
    lo = PRE - n0; out[0, lo:PRE + len(v) - n0] = np.clip(z, -30, 30); out[1, lo:PRE + len(v) - n0] = np.clip(e, -30, 30); out[2, lo:PRE + len(v) - n0] = 1; out[1, lo:lo + q] = 0
    return out, s % 2, meta
res = Parallel(n_jobs=-1, batch_size=128)(delayed(one)(s) for s in range(N))
X = np.stack([r[0] for r in res]); y = np.array([r[1] for r in res], np.int8); keep = y >= 0
M = np.stack([r[2] for r in res]); np.savez(OUT, X=X[keep], y=y[keep], M=M[keep]); print(OUT, X[keep].shape, "dropped", int((~keep).sum()))
