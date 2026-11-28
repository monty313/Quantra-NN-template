"""Persistence baseline: predict next 5m direction = prev5m_dir.

prev5m_dir is this completed 5m vs the previous completed 5m.
y_5m_dir is the next closed 5m vs this one. Not the same column.
When prev5m_dir is 0, fall back to the train class prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from holygrail.config import N_CLASSES
from holygrail.policy import encode_y


class PersistencePolicy:
    name = "persistence"

    def __init__(self, col: str = "prev5m_dir"):
        self.col = col
        self.p_continue_ = 0.5
        self.prior_ = np.full(N_CLASSES, 0.5, dtype=np.float64)

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        y = np.asarray(y, dtype=np.float64)
        prev = np.asarray(X[self.col], dtype=np.float64)
        enc = encode_y(y)
        n = enc.size
        counts = np.bincount(enc, minlength=N_CLASSES).astype(np.float64)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            counts = np.array([sw[enc == k].sum() for k in range(N_CLASSES)], dtype=np.float64)
            n = float(sw.sum())
        self.prior_ = counts / max(n, 1.0)
        nz = prev != 0
        if sample_weight is None:
            if np.any(nz):
                self.p_continue_ = float(np.mean(y[nz] == prev[nz]))
        else:
            sw = np.asarray(sample_weight, dtype=np.float64)
            w = sw[nz]
            if w.sum() > 0:
                self.p_continue_ = float(np.average((y[nz] == prev[nz]).astype(np.float64), weights=w))
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        prev = np.asarray(X[self.col], dtype=np.float64)
        n = len(prev)
        out = np.tile(self.prior_, (n, 1))
        pc = float(np.clip(self.p_continue_, 1e-6, 1.0 - 1e-6))
        # columns [-1, +1]: if prev=+1, P(+1)=p_continue; if prev=-1, P(+1)=1-p_continue
        up = prev == 1
        dn = prev == -1
        out[up, 0] = 1.0 - pc
        out[up, 1] = pc
        out[dn, 0] = pc
        out[dn, 1] = 1.0 - pc
        return out

    def explain(self, X) -> dict:
        return {
            "kind": "persistence",
            "col": self.col,
            "p_continue": self.p_continue_,
            "prior": self.prior_.tolist(),
        }
