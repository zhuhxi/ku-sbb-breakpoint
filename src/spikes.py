"""Separate pasted-on decoy spikes from genuine large values using float bits, then clean them out.

The generator builds v_t = v_{t-1} + delta_t, so v_t sits on the precision grid of v_{t-1}.
A spike pasted on afterwards is NOT used as the next value's operand: the next value's grid
ignores it. Genuine large values (heavy-tailed noise) ARE used. Needs unrescaled series.
"""
import numpy as np
import pandas as pd
from scipy import stats

from bits import bit_frame


def spike_features(path):
    d = bit_frame(path)
    out = {}
    for sid, g in d.groupby("id", sort=False):
        v = g["value"].to_numpy()
        E, eprev = g["E"].to_numpy(), g["eprev"].to_numpy()
        n0 = int((g["period"] == 0).sum())
        med = np.median(v[:n0])
        s = 1.4826 * np.median(np.abs(v[:n0] - med)) + 1e-12
        z = np.abs(v - med) / s
        out_idx = np.where(z[:-1] > 4)[0]
        # next value used the outlier as operand -> genuine; grid far below the outlier -> pasted.
        pasted = np.array([E[i + 1] - eprev[i + 1] < -1 for i in out_idx], bool)
        f = {}
        for tag, lo, hi in (("b", 0, n0), ("a", n0, len(v))):
            m = (out_idx >= lo) & (out_idx < hi)
            f[f"pasted_{tag}"] = pasted[m].sum() / (hi - lo) * 100
            f[f"genuine_{tag}"] = (~pasted[m]).sum() / (hi - lo) * 100
        f["pasted_d"], f["genuine_d"] = f["pasted_a"] - f["pasted_b"], f["genuine_a"] - f["genuine_b"]
        # clean: replace pasted spikes by the mean of their neighbours, recompute key contrasts
        c = v.copy()
        for i in out_idx[pasted]:
            c[i] = 0.5 * (c[i - 1] + c[i + 1]) if i > 0 else c[i + 1]
        x0, x1 = c[:n0], c[n0:]
        dx0, dx1 = np.diff(x0), np.diff(x1)
        f["cl_lrstd"] = np.log((x1.std() + 1e-12) / (x0.std() + 1e-12))
        f["cl_dkurt"] = stats.kurtosis(x1) - stats.kurtosis(x0)
        f["cl_dlev"] = -np.log10(max(stats.levene(dx0, dx1, center="median").pvalue, 1e-300))
        f["cl_t"] = -np.log10(max(stats.ttest_ind(x0, x1, equal_var=False).pvalue, 1e-300))
        f["cl_ks"] = stats.ks_2samp(x0, x1).statistic
        f["cl_ad"] = stats.anderson_ksamp([x0, x1], method=None).statistic
        out[sid] = f
    return pd.DataFrame.from_dict(out, orient="index")
