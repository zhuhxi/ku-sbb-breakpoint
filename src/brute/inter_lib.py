"""Shared pieces of the top-162 -> 91k interaction recipe (used by lgb_inter.py and final_inter.py)."""
import os

import numpy as np

P = dict(objective="binary", metric="auc", learning_rate=float(os.environ.get("LR", 0.02)), num_leaves=31, min_child_samples=40,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=int(os.environ.get("NTHREADS", 24)))
OPS = ("mul", "add", "sub", "div", "rdiv", "mulb2", "mula2")


class Inter:
    """Interaction builder whose fill values and epsilons come from training rows only."""

    def __init__(self, Xtr):
        self.med = np.nanmedian(Xtr, 0)
        self.eps = np.median(np.abs(np.where(np.isnan(Xtr), self.med, Xtr)), 0) * 1e-3 + 1e-9
        k = Xtr.shape[1]
        self.ia, self.ib = np.triu_indices(k, 1)
        self.names = [f"{o}__{i}__{j}" for o in OPS for i, j in zip(self.ia, self.ib)] + [f"sq__{i}" for i in range(k)]

    def _op(self, X, o, i, j):
        a, b = X[:, i], X[:, j]
        if o == "mul": return a * b
        if o == "add": return a + b
        if o == "sub": return a - b
        if o == "div": return a / (np.abs(b) + self.eps[j]) * np.sign(b + 1e-12)
        if o == "rdiv": return b / (np.abs(a) + self.eps[i]) * np.sign(a + 1e-12)
        if o == "mulb2": return a * b * b
        if o == "mula2": return a * a * b

    def fill(self, X):
        return np.where(np.isnan(X), self.med, X).astype(np.float32)

    def full(self, X):
        X = self.fill(X)
        M = np.concatenate([X] + [self._op(X, o, self.ia, self.ib) for o in OPS] + [X * X], 1)
        return np.clip(M, -1e30, 1e30, out=M)

    def select(self, X, cols):
        """Only the given column names (base columns are ints, interactions 'op__i__j' / 'sq__i')."""
        X = self.fill(X)
        out = []
        for c in cols:
            if isinstance(c, (int, np.integer)):
                out.append(X[:, c])
            elif c.startswith("sq__"):
                i = int(c.split("__")[1]); out.append(X[:, i] ** 2)
            else:
                o, i, j = c.split("__"); out.append(self._op(X, o, int(i), int(j)))
        return np.clip(np.stack(out, 1), -1e30, 1e30)
