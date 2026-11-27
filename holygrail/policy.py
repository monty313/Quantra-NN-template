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

    Default names are the Stage-1 structure heads (next closed 5m / 30m).
    The MLP policy may itself be two-headed; DualHead still wraps two copies
    for logistic / LightGBM / majority.
    """

    def __init__(self, make_policy, ltf_key: str = "y_5m", htf_key: str = "y_30m"):
        self.ltf_key = ltf_key
        self.htf_key = htf_key
        self.head_ltf = make_policy()
        self.head_htf = make_policy()
        # Old names kept as aliases so explain scripts do not fork.
        self.head_15m = self.head_ltf
        self.head_1h = self.head_htf

    def fit(self, X, y_ltf, y_htf, X_val=None, y_ltf_val=None, y_htf_val=None, sample_weight=None):
        self.head_ltf.fit(X, y_ltf, sample_weight=sample_weight, X_val=X_val, y_val=y_ltf_val)
        self.head_htf.fit(X, y_htf, sample_weight=sample_weight, X_val=X_val, y_val=y_htf_val)
        return self

    def predict_proba(self, X) -> dict[str, np.ndarray]:
        return {
            self.ltf_key: self.head_ltf.predict_proba(X),
            self.htf_key: self.head_htf.predict_proba(X),
        }

    def explain(self, X) -> dict[str, dict]:
        return {
            self.ltf_key: self.head_ltf.explain(X),
            self.htf_key: self.head_htf.explain(X),
        }
