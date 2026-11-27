"""Answer columns y_15m and y_1h (spec §12). Never stored in X, never scaled.

Why separate: if y sits in the same frame as X, it is too easy to scale it or
fit on it (a Quantra/sklearn pipeline habit). Answers describe what happened
after t; they are not features.

Missing candle at t+h is NaN, not 0. A weekend gap is not a "flat" market.
Threshold uses the same 5m ATR14 as the features, at t, so labels and X share
one volatility scale without putting ATR itself in X.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from holygrail.config import LABEL_ATR_FRACTION, LABEL_HORIZONS_MIN
from holygrail.features import FeatureTable
from holygrail.io import typical_price


@dataclass
class LabelTable:
    y_15m: np.ndarray  # float, NaN when missing
    y_1h: np.ndarray
    label_valid_15m: np.ndarray
    label_valid_1h: np.ndarray
    threshold: np.ndarray  # 0.5 * ATR14_5m at t


def build_labels(m1: pd.DataFrame, features: FeatureTable) -> LabelTable:
    """TP(x) = typical of the 1m candle closing at x. Missing if no candle at t+h."""
    tp = typical_price(m1).to_numpy(dtype=np.float64)
    open_index = m1.index
    # Candle closing at t+h has open time = this row's open + h minutes
    threshold = LABEL_ATR_FRACTION * features.atr_5m
    atr_ok = np.isfinite(threshold) & (threshold > 0)

    def one_horizon(h: int) -> tuple[np.ndarray, np.ndarray]:
        future_open = open_index + pd.Timedelta(minutes=h)
        loc = open_index.get_indexer(future_open)
        have = loc >= 0
        idx = np.arange(len(m1))
        move = np.full(len(m1), np.nan, dtype=np.float64)
        move[have] = tp[loc[have]] - tp[idx[have]]
        y = np.full(len(m1), np.nan, dtype=np.float64)
        ok = have & atr_ok
        m = move[ok]
        th = threshold[ok]
        labels = np.zeros(m.shape, dtype=np.float64)
        labels = np.where(m >= th, 1.0, labels)
        labels = np.where(m <= -th, -1.0, labels)
        y[ok] = labels
        return y, ok

    y15, v15 = one_horizon(LABEL_HORIZONS_MIN[0])
    y60, v60 = one_horizon(LABEL_HORIZONS_MIN[1])
    return LabelTable(
        y_15m=y15,
        y_1h=y60,
        label_valid_15m=v15,
        label_valid_1h=v60,
        threshold=threshold,
    )
