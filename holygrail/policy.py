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


class MultiHead:
    """Independent 3-class heads sharing one policy class, not one net.

    Default keys: next-closed 5m / 30m / 1h structure.
    """

    def __init__(self, make_policy, keys: tuple[str, ...] = ("y_5m", "y_30m", "y_1h_structure")):
        self.keys = list(keys)
        self.heads = {k: make_policy() for k in self.keys}
        self.head_ltf = self.heads[self.keys[0]]
        self.head_htf = self.heads[self.keys[1]] if len(self.keys) > 1 else self.head_ltf
        self.head_1h = self.heads.get("y_1h_structure", self.head_htf)
        self.head_15m = self.head_ltf

    def fit(self, X, ys: dict, X_val=None, ys_val=None, sample_weight=None):
        for k, head in self.heads.items():
            yv = None if ys_val is None else ys_val[k]
            head.fit(X, ys[k], sample_weight=sample_weight, X_val=X_val, y_val=yv)
        return self

    def predict_proba(self, X) -> dict[str, np.ndarray]:
        return {k: h.predict_proba(X) for k, h in self.heads.items()}

    def explain(self, X) -> dict[str, dict]:
        return {k: h.explain(X) for k, h in self.heads.items()}


# Back-compat name used in earlier Stage-1 notes.
DualHead = MultiHead
