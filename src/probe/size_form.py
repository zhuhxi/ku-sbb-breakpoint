"""Three-bin uniform break sizes (best2 form) vs continuous (half-normal / exponential): same budget of random trials per form,
same objective on the BREAK class only (no-break is unaffected by sizes): C2ST AUC on the 12 statistics of probe/gen_eval2.py
(+ mean AD / 100 as tie-breaker). Then the best of each form is re-scored on different fake seeds.
    python probe/size_form.py TRIALS"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "probe")
import gen_eval2 as G
t0 = time.time(); NT = int(sys.argv[1]); R = G.real(); base = G.parse("best2"); rng = np.random.default_rng(0)
G.NF = 2000


def score(P, seed=700_000):
    r = G.evaluate1(P, R, seed); return r["c2st"] + r["ad"] / 100, r


def ev1(P, R, seed):
    import lightgbm as lgb
    from joblib import Parallel, delayed
    from scipy.stats import anderson_ksamp, ks_2samp
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    F = pd.DataFrame(Parallel(n_jobs=30, batch_size=32)(delayed(lambda s: G.stats(*G.gen(P, seed + 2 * s + 1, True)[:2]))(s) for s in range(G.NF))).replace([np.inf, -np.inf], np.nan)
    r = R[R.y == 1].drop(columns="y").replace([np.inf, -np.inf], np.nan)
    ad = np.mean([anderson_ksamp([r[c].dropna().to_numpy(), F[c].dropna().to_numpy()]).statistic for c in r])
    ks = np.mean([ks_2samp(r[c].dropna(), F[c].dropna()).statistic for c in r])
    X = pd.concat([r, F]); t = np.r_[np.zeros(len(r)), np.ones(len(F))]
    auc = roc_auc_score(t, cross_val_predict(lgb.LGBMClassifier(n_estimators=150, learning_rate=.05, num_leaves=15, verbose=-1, n_jobs=30), X, t, cv=5, method="predict_proba")[:, 1])
    return dict(c2st=auc, ad=ad, ks=ks)


G.evaluate1 = ev1
U = lambda a, b: float(rng.uniform(a, b))
forms = {
    "three-bin": lambda: dict(base, lvl_big=U(.05, .35), lvl_big_v=(U(1, 1.6), U(2, 4)), lvl_mid=U(.1, .35), lvl_weak=(.01, U(.3, .7)),
                              lvl_delay=U(0, .3), spread_ln2=U(.1, .45), spread_small=(.01, U(.25, .5)), dep_big=U(.05, .3)),
    "half-normal": lambda: dict(base, size_mode="hn", lvl_cs=U(.3, 2), spread_cs=U(.1, .6), dep_cs=U(.05, .5), lvl_delay=U(0, .3)),
    "exponential": lambda: dict(base, size_mode="exp", lvl_cs=U(.2, 1.5), spread_cs=U(.05, .5), dep_cs=U(.05, .4), lvl_delay=U(0, .3)),
}
best = {}
for fm, draw in forms.items():
    trials = [base] if fm == "three-bin" else []
    trials += [draw() for _ in range(NT)]
    sc = [score(P) for P in trials]
    i = int(np.argmin([s[0] for s in sc])); best[fm] = trials[i]
    print(f"{fm:12s} best of {len(trials)}: C2ST {sc[i][1]['c2st']:.4f} AD {sc[i][1]['ad']:.2f} KS {sc[i][1]['ks']:.4f}"
          + (f"  (best2 itself: C2ST {sc[0][1]['c2st']:.4f} AD {sc[0][1]['ad']:.2f})" if fm == "three-bin" else "") + f"  {time.time() - t0:.0f}s", flush=True)
print("re-scored on new fake seeds (3 seed sets, mean):")
for fm, P in list(best.items()) + [("best2 as is", base)]:
    rs = [ev1(P, R, sd) for sd in (810_000, 820_000, 830_000)]
    print(f"  {fm:12s} C2ST {np.mean([r['c2st'] for r in rs]):.4f} AD {np.mean([r['ad'] for r in rs]):.2f} KS {np.mean([r['ks'] for r in rs]):.4f}  "
          + {k: (round(v, 3) if isinstance(v, float) else v) for k, v in P.items() if k in ("size_mode", "lvl_cs", "spread_cs", "dep_cs", "lvl_delay", "lvl_big", "lvl_mid", "spread_ln2", "dep_big")}.__repr__(), flush=True)
import json
for fm, P in best.items():
    json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()}, open(f"outputs/factory/ku_gen_size_{fm}.json", "w"), indent=1)
print(f"{time.time() - t0:.0f}s")
