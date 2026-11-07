"""Expert checklist on the model's worst errors (combo10g OOF). For the 100 worst false alarms, the 100 worst missed
breaks, and 100 correctly confident cases of each label, tally what an expert looks for in a plot: a big spike after
(or before) the boundary, its side (up / down), a slow drift after the boundary, a visible step, a visible change in
wiggle size. Plots of 24 false alarms and 24 missed breaks for the eye check.

    python probe/miss_expert.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def check(v, n0):
    a, b = v[:n0], v[n0:]
    med, mad = np.median(a), np.median(np.abs(a - np.median(a))) * 1.4826 + 1e-12
    za, zb = (a - med) / mad, (b - med) / mad
    t = np.arange(len(b))
    slope = np.polyfit(t, b, 1)[0] * len(b) / mad                  # rise over the after part, in robust sd
    return {"spike after (|z|>5)": (np.abs(zb) > 5).any(), "spike before (|z|>5)": (np.abs(za) > 5).any(),
            "spikes after - before (per 100 pts)": 100 * ((np.abs(zb) > 4).mean() - (np.abs(za) > 4).mean()),
            "biggest spike after is downward": zb[np.argmax(np.abs(zb))] < 0,
            "drift after (rise > 1 sd)": abs(slope) > 1,
            "visible step (median shift > 0.5 sd)": abs(np.median(b) - med) / mad > .5,
            "visible wiggle change (MAD ratio > 1.3)": max(np.median(np.abs(b - np.median(b))) * 1.4826 / mad,
                                                         mad / (np.median(np.abs(b - np.median(b))) * 1.4826 + 1e-12)) > 1.3}


lab = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
oof = pd.Series(np.load("outputs/factory/confirm_combo10g.npz")["oof"], index=lab.index)
S = pd.read_parquet("data/train_series.parquet").sort_values(["id", "time"])
groups = {"FALSE alarm": oof[lab == 0].nlargest(100).index, "correct no-break": oof[lab == 0].nsmallest(100).index,
          "MISSED break": oof[lab == 1].nsmallest(100).index, "caught break": oof[lab == 1].nlargest(100).index}
need = set().union(*map(set, groups.values()))
G = {i: (g["value"].to_numpy(), int((g["period"] == 0).sum())) for i, g in S[S.id.isin(need)].groupby("id")}
T = pd.DataFrame({k: pd.DataFrame([check(*G[i]) for i in ids]).mean() for k, ids in groups.items()})
pct = [r for r in T.index if "per 100" not in r]
T.loc[pct] *= 100
print("share of series with each sign (%), spike rate difference in spikes per 100 points")
print(T.round(1).to_string())
for name in ("FALSE alarm", "MISSED break"):
    fig, ax = plt.subplots(6, 4, figsize=(18, 15))
    for a, i in zip(ax.ravel(), groups[name][:24]):
        v, n0 = G[i]
        a.plot(v, lw=.5, color="0.4")
        a.plot(pd.Series(v).rolling(30, center=True).median(), color="tab:blue", lw=1.2)
        a.axvline(n0, color="r", ls="--", lw=.8)
        a.set_title(f"{name} oof {oof[i]:.2f} n={len(v)}", fontsize=8)
        a.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(f"outputs/probe/expert_{name.split()[0].lower()}.png", dpi=70)


def near(v, n0, w=10):                                            # the biggest |z| within w points after the boundary
    a = v[:n0]
    med, mad = np.median(a), np.median(np.abs(a - np.median(a))) * 1.4826 + 1e-12
    z = np.abs(v - med) / mad
    return z[n0:n0 + w].max() > 4, z[n0 - w:n0].max() > 4, z[n0 + w:].max() > 4 and (z[n0 + w:] > 4).mean() < .002


R = pd.DataFrame({k: pd.DataFrame([near(*G[i]) for i in ids], columns=["spike in first 10 after", "spike in last 10 before",
                                                                    "spike only later, rare"]).mean() * 100 for k, ids in groups.items()})
print("\nspike right at the boundary (|z| > 4), % of series")
print(R.round(1).to_string())


def lasting(v, n0):                                              # is the change still there in the last half after?
    a, b = v[:n0], v[n0:]
    med, mad = np.median(a), np.median(np.abs(a - np.median(a))) * 1.4826 + 1e-12
    h = len(b) // 2
    m = lambda s: np.median(np.abs(s - np.median(s))) * 1.4826 / mad
    s1, s2 = (np.median(b[:h]) - med) / mad, (np.median(b[h:]) - med) / mad
    w1, w2 = np.log(m(b[:h])), np.log(m(b[h:]))
    return {"step early half": abs(s1), "step late half": abs(s2), "step kept (same sign, late >= half of early)": (abs(s1) > .3) and s1 * s2 > 0 and abs(s2) >= .5 * abs(s1),
            "wiggle early half": abs(w1), "wiggle late half": abs(w2),
            "wiggle change kept": (abs(w1) > .2) and w1 * w2 > 0 and abs(w2) >= .5 * abs(w1)}


Q = pd.DataFrame({k: pd.DataFrame([lasting(*G[i]) for i in ids]).mean() for k, ids in groups.items()})
Q.loc[[r for r in Q.index if "kept" in r]] *= 100
print("\nis the change lasting? (sizes in robust sd / log ratio; 'kept' rows in % of series)")
print(Q.round(2).to_string())
