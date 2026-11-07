"""Sanity of the break mechanism per family (retuned three-bin generator, break class): how big is the change the series really shows?
level: |post mean - pre mean| / pre sd of the SERIES; spread: |log(post sd / pre sd)| of the first differences (series level);
dep: |acf1 post - acf1 pre| of the differences; shape: pre/post noise type identical (no real change)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "probe"); import gen_eval2 as G, ku_gen as K
P = G.parse("outputs/factory/ku_gen_size_three-bin.json"); rows = []
ac = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]
for s in range(6000):
    x, n0, h = K.gen(P, 5_000_000 + s, True); pre, post = x[:n0], x[n0:]; dp, dq = np.diff(pre), np.diff(post)
    rows.append(dict(fam=h["family"], kind=h["kind"], lvl=abs(post.mean() - pre.mean()) / pre.std(),
                     lvl_d=abs(h["lvl_d"]), spr=abs(np.log(dq.std() / dp.std())), dep=abs(ac(dq) - ac(dp))))
D = pd.DataFrame(rows)
for k, col in (("level", "lvl"), ("spread", "spr"), ("dep", "dep")):
    T = D[D.kind == k].groupby("fam")[col].agg(["count", "median", lambda v: (v < .1).mean()]).round(3)
    T.columns = ["n", f"median {col}", "share <.1 (invisible)"]; print(f"--- {k} break: change visible in the series\n{T}")
print("--- median generated level size |lvl_d| (noise-sd units) by family:\n", D[D.kind == "level"].groupby("fam").lvl_d.median().round(3).to_dict())
# shape: how often is the post noise the same type as the pre noise?
same = []
for s in range(4000):
    r = np.random.default_rng(s); P2 = dict(P); K.noise(r, 500, P2); a = dict(K._LAST); K.noise(r, 500, P2); b = dict(K._LAST)
    same.append(a["ntype"] == b["ntype"] and (a["ntype"] != 1 or abs(1 / a["ndf"] - 1 / b["ndf"]) < .03) and a["garch"] == b["garch"])
print(f"--- shape break (redraw): post noise same type (+ df within 1/df .03, same GARCH) = {np.mean(same):.3f}")
