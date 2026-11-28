"""Answer columns. Never stored in X, never scaled.

Why separate: if y sits in the same frame as X, it is too easy to scale it or
fit on it (a Quantra/sklearn pipeline habit). Answers describe what happened
after t; they are not features.

Primary Stage-1 heads: next **closed** 5m, 30m, and 1h **direction**
(sign of typical-price change). No ATR size fence. Std / ATR / RSI stay in X
only. Exact-zero typical change is invalid (drop), not class 0.

For each F-bar family (shared across every 1m inside the current F-bar):
- now = typical price of **this** F-bar when it has closed (B_t). Labels may
  use that future close; X at early 1m rows does not (developing only).
- future = typical price of the next F-bar (B_{t+1}). X never contains B_{t+1}.
- y = sign(future − now) ∈ {−1, +1}
- Missing this/next F-bar, or future == now → NaN, not 0.

We do **not** use B_{t-1} vs B_{t+1} (skipping the bar X is watching).

Old y_15m / y_1h stay as unused comparison columns (cheap). Those still use
the old 0.5× ATR tick rule so the mismatch can be compared; they are not fit.

Missing candle at t+h is NaN, not 0. A weekend gap is not a "flat" market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from holygrail.config import (
    LABEL_ATR_FRACTION,
    LABEL_HORIZONS_MIN,
    STRUCTURE_BAR_MINUTES,
    TF_FREQ,
)
from holygrail.features import FeatureTable
from holygrail.io import typical_price
from holygrail.timing import resample_ohlc


@dataclass
class LabelTable:
    y_15m: np.ndarray  # comparison only
    y_1h: np.ndarray  # comparison only
    label_valid_15m: np.ndarray
    label_valid_1h: np.ndarray
    threshold: np.ndarray  # 0.5 * ATR14_5m at t (old heads only)
    y_5m: np.ndarray
    y_5m_dir: np.ndarray  # alias of y_5m; same array, never a second definition
    y_5m_nextclose_vs_typ: np.ndarray  # Round 3: 1/0, never a feature
    y_5m_nextclose_vs_close: np.ndarray  # Round 5 L2: next close vs this close
    y_5m_close2_vs_typ: np.ndarray  # Round 4 A: close(t+2) vs typical(t)
    y_5m_close2_vs_close: np.ndarray  # Round 4 B: close(t+2) vs close(t)
    y_30m: np.ndarray
    y_1h_structure: np.ndarray
    label_valid_5m: np.ndarray
    label_valid_nextclose_vs_typ: np.ndarray
    label_valid_nextclose_vs_close: np.ndarray
    label_valid_close2_vs_typ: np.ndarray
    label_valid_close2_vs_close: np.ndarray
    label_valid_30m: np.ndarray
    label_valid_1h_structure: np.ndarray


def _ternary(move: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Old tick-head fence. Not used for training heads."""
    labels = np.zeros(move.shape, dtype=np.float64)
    labels = np.where(move >= th, 1.0, labels)
    labels = np.where(move <= -th, -1.0, labels)
    return labels


def structure_direction(
    m1: pd.DataFrame,
    tf_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared next-closed-F direction y for every 1m row.

    y = sign(TP(B_{t+1}) − TP(B_t)). Exact 0 and missing bars are invalid.
    Returns (y, valid). y is NaN where valid is False. No ATR in this label.
    """
    freq = TF_FREQ[tf_key]
    bar_minutes = STRUCTURE_BAR_MINUTES[tf_key]
    completed = resample_ohlc(m1, freq)
    tp = (completed["high"] + completed["low"] + completed["close"]) / 3.0
    tp_v = tp.to_numpy(dtype=np.float64)
    idx = completed.index

    current_open = m1.index.floor(freq)
    delta = pd.Timedelta(minutes=bar_minutes)
    next_open = current_open + delta

    loc_now = idx.get_indexer(current_open)
    loc_next = idx.get_indexer(next_open)
    have = (loc_now >= 0) & (loc_next >= 0)

    y = np.full(len(m1), np.nan, dtype=np.float64)
    valid = np.zeros(len(m1), dtype=bool)
    if have.any():
        move = tp_v[loc_next[have]] - tp_v[loc_now[have]]
        nonzero = move != 0.0
        ok = np.zeros(have.sum(), dtype=bool)
        ok[nonzero] = True
        y_have = np.full(have.sum(), np.nan, dtype=np.float64)
        y_have[nonzero] = np.sign(move[nonzero])
        y[have] = y_have
        valid[have] = ok
    return y, valid


def nextclose_vs_typ(m1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Round 3 label. Not a feature. 1/0, ties dropped.

    At the close of this 5m bar: 1 if next 5m close > this 5m typical,
    0 if next 5m close < this 5m typical. Typical = (H+L+C)/3 of this bar.
    Missing next bar or exact equality → NaN / invalid (not class 0).
    Shared by every 1m inside the current 5m, same as other structure heads.
    """
    freq = TF_FREQ["5m"]
    bar_minutes = STRUCTURE_BAR_MINUTES["5m"]
    completed = resample_ohlc(m1, freq)
    tp = (completed["high"] + completed["low"] + completed["close"]) / 3.0
    close = completed["close"]
    idx = completed.index
    current_open = m1.index.floor(freq)
    next_open = current_open + pd.Timedelta(minutes=bar_minutes)
    loc_now = idx.get_indexer(current_open)
    loc_next = idx.get_indexer(next_open)
    have = (loc_now >= 0) & (loc_next >= 0)
    y = np.full(len(m1), np.nan, dtype=np.float64)
    valid = np.zeros(len(m1), dtype=bool)
    if have.any():
        move = close.to_numpy(dtype=np.float64)[loc_next[have]] - tp.to_numpy(dtype=np.float64)[loc_now[have]]
        nonzero = move != 0.0
        y_have = np.full(have.sum(), np.nan, dtype=np.float64)
        y_have[nonzero] = np.where(move[nonzero] > 0.0, 1.0, 0.0)
        y[have] = y_have
        ok = np.zeros(have.sum(), dtype=bool)
        ok[nonzero] = True
        valid[have] = ok
    return y, valid


def nextclose_vs_close(m1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Round 5 L2. 1 if next 5m close > this 5m close. Not a feature."""
    return _close_ahead_vs_ref(m1, n_bars=1, ref="close")


def _close_ahead_vs_ref(
    m1: pd.DataFrame,
    *,
    n_bars: int,
    ref: str,
) -> tuple[np.ndarray, np.ndarray]:
    """1 if close of bar t+n_bars > ref of bar t, else 0. Ties / missing → NaN.

    ref: "typ" = (H+L+C)/3 of bar t; "close" = close of bar t.
    Shared by every 1m inside the current 5m. Never a feature.
    """
    if ref not in ("typ", "close"):
        raise ValueError(ref)
    freq = TF_FREQ["5m"]
    bar_minutes = STRUCTURE_BAR_MINUTES["5m"]
    completed = resample_ohlc(m1, freq)
    tp = (completed["high"] + completed["low"] + completed["close"]) / 3.0
    close = completed["close"]
    ref_s = tp if ref == "typ" else close
    idx = completed.index
    current_open = m1.index.floor(freq)
    fut_open = current_open + pd.Timedelta(minutes=bar_minutes * n_bars)
    loc_now = idx.get_indexer(current_open)
    loc_fut = idx.get_indexer(fut_open)
    have = (loc_now >= 0) & (loc_fut >= 0)
    y = np.full(len(m1), np.nan, dtype=np.float64)
    valid = np.zeros(len(m1), dtype=bool)
    if have.any():
        move = close.to_numpy(dtype=np.float64)[loc_fut[have]] - ref_s.to_numpy(dtype=np.float64)[loc_now[have]]
        nonzero = move != 0.0
        y_have = np.full(have.sum(), np.nan, dtype=np.float64)
        y_have[nonzero] = np.where(move[nonzero] > 0.0, 1.0, 0.0)
        y[have] = y_have
        ok = np.zeros(have.sum(), dtype=bool)
        ok[nonzero] = True
        valid[have] = ok
    return y, valid


def close2_vs_typ(m1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Round 4 A. 1 if close of 5m bar t+2 > typical of bar t. Not a feature."""
    return _close_ahead_vs_ref(m1, n_bars=2, ref="typ")


def close2_vs_close(m1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Round 4 B. 1 if close of 5m bar t+2 > close of bar t. Not a feature."""
    return _close_ahead_vs_ref(m1, n_bars=2, ref="close")


def build_labels(m1: pd.DataFrame, features: FeatureTable) -> LabelTable:
    """TP(x) = typical of the 1m candle closing at x. Missing if no candle at t+h.

    Training heads: next-closed 5m / 30m / 1h direction (shared per F-bar).
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
    y5, v5 = structure_direction(m1, "5m")
    y_ncvt, v_ncvt = nextclose_vs_typ(m1)
    y_ncc, v_ncc = nextclose_vs_close(m1)
    y_a, v_a = close2_vs_typ(m1)
    y_b, v_b = close2_vs_close(m1)
    y30, v30 = structure_direction(m1, "30m")
    y1h_s, v1h_s = structure_direction(m1, "1h")
    return LabelTable(
        y_15m=y15,
        y_1h=y60,
        label_valid_15m=v15,
        label_valid_1h=v60,
        threshold=threshold,
        y_5m=y5,
        y_5m_dir=y5,  # alias: identical object, not recomputed
        y_5m_nextclose_vs_typ=y_ncvt,
        y_5m_nextclose_vs_close=y_ncc,
        y_5m_close2_vs_typ=y_a,
        y_5m_close2_vs_close=y_b,
        y_30m=y30,
        y_1h_structure=y1h_s,
        label_valid_5m=v5,
        label_valid_nextclose_vs_typ=v_ncvt,
        label_valid_nextclose_vs_close=v_ncc,
        label_valid_close2_vs_typ=v_a,
        label_valid_close2_vs_close=v_b,
        label_valid_30m=v30,
        label_valid_1h_structure=v1h_s,
    )
