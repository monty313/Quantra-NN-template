"""Answer columns. Never stored in X, never scaled.

Why separate: if y sits in the same frame as X, it is too easy to scale it or
fit on it (a Quantra/sklearn pipeline habit). Answers describe what happened
after t; they are not features.

Primary Stage-1 heads: next **closed** 5m, 30m, and 1h structure direction.
X is live_1m + developing 5m/30m/1h — a 15m 1m-tick vs 5m ATR asks a
different question than the features describe.

For each F-bar family (shared across every 1m inside the current F-bar):
- now = typical price of **this** F-bar when it has closed (B_t). Labels may
  use that future close; X at early 1m rows does not (developing only).
- future = typical price of the next F-bar (B_{t+1}). X never contains B_{t+1}.
- threshold = 0.5 × ATR14 of that same F-bar series, at B_t (spec fraction;
  same-HTF ATR as the horizon).
- Missing this/next F-bar (weekend hole, tail) → NaN, not 0.

We do **not** use B_{t-1} vs B_{t+1} (skipping the bar X is watching). That
made class 0 a majority mush again. Next-bar vs this-bar TP is the structure
question the developing columns describe.

Old y_15m / y_1h stay as unused comparison columns (cheap).

Missing candle at t+h is NaN, not 0. A weekend gap is not a "flat" market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from holygrail.config import (
    ATR_PERIOD,
    LABEL_ATR_FRACTION,
    LABEL_HORIZONS_MIN,
    STRUCTURE_ATR_FRACTION,
    STRUCTURE_BAR_MINUTES,
    STRUCTURE_WIDE_ATR_FRACTION,
    TF_FREQ,
)
from holygrail.features import FeatureTable
from holygrail.indicators import atr_wilder
from holygrail.io import typical_price
from holygrail.timing import resample_ohlc


@dataclass
class LabelTable:
    y_15m: np.ndarray  # comparison only
    y_1h: np.ndarray  # comparison only
    label_valid_15m: np.ndarray
    label_valid_1h: np.ndarray
    threshold: np.ndarray  # 0.5 * ATR14_5m at t (old heads)
    y_5m: np.ndarray
    y_30m: np.ndarray
    y_1h_structure: np.ndarray
    label_valid_5m: np.ndarray
    label_valid_30m: np.ndarray
    label_valid_1h_structure: np.ndarray
    threshold_5m: np.ndarray
    threshold_30m: np.ndarray
    threshold_1h: np.ndarray
    y_5m_wide: np.ndarray  # 1.0× ATR extra lever
    y_30m_wide: np.ndarray
    label_valid_5m_wide: np.ndarray
    label_valid_30m_wide: np.ndarray


def _ternary(move: np.ndarray, th: np.ndarray) -> np.ndarray:
    labels = np.zeros(move.shape, dtype=np.float64)
    labels = np.where(move >= th, 1.0, labels)
    labels = np.where(move <= -th, -1.0, labels)
    return labels


def structure_direction(
    m1: pd.DataFrame,
    tf_key: str,
    atr_fraction: float = STRUCTURE_ATR_FRACTION,
    atr_period: int = ATR_PERIOD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared next-closed-F structure y for every 1m row.

    Returns (y, valid, threshold). y is NaN where valid is False.
    """
    freq = TF_FREQ[tf_key]
    bar_minutes = STRUCTURE_BAR_MINUTES[tf_key]
    completed = resample_ohlc(m1, freq)
    tp = (completed["high"] + completed["low"] + completed["close"]) / 3.0
    atr = atr_wilder(
        completed["high"].to_numpy(),
        completed["low"].to_numpy(),
        completed["close"].to_numpy(),
        atr_period,
    )
    tp_v = tp.to_numpy(dtype=np.float64)
    idx = completed.index

    current_open = m1.index.floor(freq)
    delta = pd.Timedelta(minutes=bar_minutes)
    next_open = current_open + delta

    loc_now = idx.get_indexer(current_open)
    loc_next = idx.get_indexer(next_open)
    have = (loc_now >= 0) & (loc_next >= 0)
    th_now = np.full(len(m1), np.nan, dtype=np.float64)
    th_now[have] = atr_fraction * atr[loc_now[have]]
    atr_ok = np.isfinite(th_now) & (th_now > 0)
    ok = have & atr_ok

    y = np.full(len(m1), np.nan, dtype=np.float64)
    if ok.any():
        move = tp_v[loc_next[ok]] - tp_v[loc_now[ok]]
        y[ok] = _ternary(move, th_now[ok])
    return y, ok, th_now


def build_labels(m1: pd.DataFrame, features: FeatureTable) -> LabelTable:
    """TP(x) = typical of the 1m candle closing at x. Missing if no candle at t+h.

    Also builds next-closed 5m / 30m / 1h structure heads (shared per F-bar).
    """
    tp = typical_price(m1).to_numpy(dtype=np.float64)
    open_index = m1.index
    threshold = LABEL_ATR_FRACTION * features.atr_5m
    atr_ok = np.isfinite(threshold) & (threshold > 0)

    def one_tick_horizon(h: int) -> tuple[np.ndarray, np.ndarray]:
        future_open = open_index + pd.Timedelta(minutes=h)
        loc = open_index.get_indexer(future_open)
        have = loc >= 0
        idx = np.arange(len(m1))
        move = np.full(len(m1), np.nan, dtype=np.float64)
        move[have] = tp[loc[have]] - tp[idx[have]]
        y = np.full(len(m1), np.nan, dtype=np.float64)
        ok = have & atr_ok
        y[ok] = _ternary(move[ok], threshold[ok])
        return y, ok

    y15, v15 = one_tick_horizon(LABEL_HORIZONS_MIN[0])
    y60, v60 = one_tick_horizon(LABEL_HORIZONS_MIN[1])
    y5, v5, th5 = structure_direction(m1, "5m", STRUCTURE_ATR_FRACTION)
    y30, v30, th30 = structure_direction(m1, "30m", STRUCTURE_ATR_FRACTION)
    y1h_s, v1h_s, th1h = structure_direction(m1, "1h", STRUCTURE_ATR_FRACTION)
    y5w, v5w, _ = structure_direction(m1, "5m", STRUCTURE_WIDE_ATR_FRACTION)
    y30w, v30w, _ = structure_direction(m1, "30m", STRUCTURE_WIDE_ATR_FRACTION)
    return LabelTable(
        y_15m=y15,
        y_1h=y60,
        label_valid_15m=v15,
        label_valid_1h=v60,
        threshold=threshold,
        y_5m=y5,
        y_30m=y30,
        y_1h_structure=y1h_s,
        label_valid_5m=v5,
        label_valid_30m=v30,
        label_valid_1h_structure=v1h_s,
        threshold_5m=th5,
        threshold_30m=th30,
        threshold_1h=th1h,
        y_5m_wide=y5w,
        y_30m_wide=y30w,
        label_valid_5m_wide=v5w,
        label_valid_30m_wide=v30w,
    )
