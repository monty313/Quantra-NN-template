"""Pluggable policy: fit(X, y) / predict_proba(X) / explain(X).

LightGBM, logistic, MLP, or a later RL head can swap without touching features.
predict_proba columns are always [-1, 0, +1] so callers never depend on
sklearn's class-order quirks.

This is the contract Stage 2 / RL should keep. Do not put entries, sizing, or
PnL inside a Policy — that was the svm-ftmo / the-truth mix that made
"the model" and "the broker bot" the same object.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int8)  # predict_proba columns


def encode_y(y: np.ndarray) -> np.ndarray:
    """Map {-1,0,+1} → {0,1,2}."""
    out = np.empty(y.shape, dtype=np.int64)
    out[y == -1] = 0
    out[y == 0] = 1
    out[y == 1] = 2
    return out


def decode_y(c: np.ndarray) -> np.ndarray:
    return CLASS_ORDER[c]


@runtime_checkable
class Policy(Protocol):
    name: str

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> Any: ...

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Shape (n, 3) with columns [-1, 0, +1]."""
        ...

    def explain(self, X: pd.DataFrame | np.ndarray) -> dict: ...


class DualHead:
    """Two independent 3-class heads sharing one policy class, not one net.

    The MLP policy may itself be two-headed; DualHead still wraps two copies
    for logistic / LightGBM / majority.
    """

    def __init__(self, make_policy):
        self.head_15m = make_policy()
        self.head_1h = make_policy()

    def fit(self, X, y_15m, y_1h, X_val=None, y_15m_val=None, y_1h_val=None, sample_weight=None):
        self.head_15m.fit(X, y_15m, sample_weight=sample_weight, X_val=X_val, y_val=y_15m_val)
        self.head_1h.fit(X, y_1h, sample_weight=sample_weight, X_val=X_val, y_val=y_1h_val)
        return self

    def predict_proba(self, X) -> dict[str, np.ndarray]:
        return {
            "y_15m": self.head_15m.predict_proba(X),
            "y_1h": self.head_1h.predict_proba(X),
        }

    def explain(self, X) -> dict[str, dict]:
        return {
            "y_15m": self.head_15m.explain(X),
            "y_1h": self.head_1h.explain(X),
        }
