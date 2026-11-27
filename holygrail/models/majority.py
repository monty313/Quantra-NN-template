"""Majority / class-prior baseline. Constant probabilities, no features used.

Uses train class frequencies, not a one-hot 'always pick the mode'.
One-hot majority looks strong on accuracy and terrible on logloss.
Stage 1 scores logloss and calibration, so the fair baseline is the prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from holygrail.policy import CLASS_ORDER, encode_y


class MajorityPolicy:
    name = "majority"

    def __init__(self):
        self.proba_ = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        y = np.asarray(y)
        enc = encode_y(y)
        n = enc.size
        counts = np.bincount(enc, minlength=3).astype(np.float64)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            counts = np.array(
                [sw[enc == k].sum() for k in range(3)], dtype=np.float64
            )
            n = sw.sum()
        self.proba_ = counts / max(n, 1.0)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        n = len(X)
        return np.tile(self.proba_, (n, 1))

    def explain(self, X) -> dict:
        return {
            "kind": "majority",
            "class_order": CLASS_ORDER.tolist(),
            "proba": self.proba_.tolist(),
            "note": "constant class priors; features unused",
        }
