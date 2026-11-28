"""MT5-matching indicator primitives (closed-bar series).

Decisions (published MT5 formulas; no live terminal in this environment):
- std / Bollinger divisor: ddof=0 (population, iStdDev / iBands)
- ATR: Wilder (iATR) — SMA of first `period` TRs, then recursive.
  pandas ewm(alpha=1/period) from bar 0 (adaptive-risk) is NOT the same seed.
- RSI: Wilder (iRSI) — SMA of first `period` changes, then recursive
- CCI: (TP - SMA) / (0.015 * MAD) with MAD over n. If MAD=0, NaN (not 0.0).
- SMA shift: bar-index offset after the SMA (iMA `shift`)

rolling_sum refuses windows that contain NaN. A cumsum of a CCI series that
starts with NaNs would poison every later SMA-of-CCI value forever.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from holygrail.config import ATR_PERIOD, BB_DEVIATION, BB_DDOF, CCI_CONSTANT, STD_DDOF


def rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Sum of the last `window` values. NaN if the window is incomplete or contains NaN."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if window <= 0:
        return np.zeros_like(x, dtype=np.float64)
    if window == 1:
        return x.copy()
    n = x.size
    if n < window:
        return out
    if np.isfinite(x).all():
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[window - 1 :] = c[window:] - c[:-window]
        return out
    w = sliding_window_view(x, window)
    good = np.isfinite(w).all(axis=1)
    s = np.where(good, w.sum(axis=1), np.nan)
    out[window - 1 :] = s
    return out


def sma(x: np.ndarray, period: int) -> np.ndarray:
    s = rolling_sum(x, period)
    return s / period


def wilder_smooth(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing with SMA seed at index period-1."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    out[period - 1] = np.mean(x[:period])
    k = 1.0 / period
    prev = out[period - 1]
    for i in range(period, n):
        prev = prev + k * (x[i] - prev)
        out[i] = prev
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    prev_c = np.empty_like(close)
    prev_c[0] = np.nan
    prev_c[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    tr[0] = high[0] - low[0]
    return tr


def atr_wilder(high, low, close, period: int = ATR_PERIOD) -> np.ndarray:
    return wilder_smooth(true_range(high, low, close), period)


def atr_simple(high, low, close, period: int = ATR_PERIOD) -> np.ndarray:
    return sma(true_range(high, low, close), period)


def rolling_std(x: np.ndarray, period: int, ddof: int = STD_DDOF) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    w = sliding_window_view(x, period)
    mean = w.mean(axis=1)
    sse = np.sum((w - mean[:, None]) ** 2, axis=1)
    denom = period - ddof
    out[period - 1 :] = np.sqrt(np.maximum(sse / denom, 0.0))
    return out


def cci(tp: np.ndarray, period: int, constant: float = CCI_CONSTANT) -> np.ndarray:
    tp = np.asarray(tp, dtype=np.float64)
    n = tp.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    w = sliding_window_view(tp, period)
    mean = w.mean(axis=1)
    mad = np.mean(np.abs(w - mean[:, None]), axis=1)
    denom = constant * mad
    val = np.full(mean.shape, np.nan)
    np.divide(tp[period - 1 :] - mean, denom, out=val, where=denom > 0)
    out[period - 1 :] = val
    return out


def rsi_wilder(price: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MT5 iRSI. Returns rsi, avg_gain, avg_loss aligned to `price`.

    First RSI is at index `period` (needs `period` changes).
    avg_gain/avg_loss are Wilder-smoothed and live on the same index as rsi.
    """
    price = np.asarray(price, dtype=np.float64)
    n = price.size
    rsi = np.full(n, np.nan, dtype=np.float64)
    ag = np.full(n, np.nan, dtype=np.float64)
    al = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return rsi, ag, al
    delta = np.diff(price)
    gains = np.clip(delta, 0.0, None)
    losses = np.clip(-delta, 0.0, None)
    ag[period] = gains[:period].mean()
    al[period] = losses[:period].mean()
    k = 1.0 / period
    for i in range(period + 1, n):
        ag[i] = ag[i - 1] + k * (gains[i - 1] - ag[i - 1])
        al[i] = al[i - 1] + k * (losses[i - 1] - al[i - 1])
    both0 = (ag == 0) & (al == 0)
    al0 = (al == 0) & (ag > 0)
    ag0 = (ag == 0) & (al > 0)
    rs = np.divide(ag, al, out=np.full_like(ag, np.nan), where=(al > 0) & np.isfinite(al))
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = np.where(both0, 50.0, rsi)
    rsi = np.where(al0, 100.0, rsi)
    rsi = np.where(ag0, 0.0, rsi)
    return rsi, ag, al


def bollinger(
    close: np.ndarray,
    period: int = 5,
    deviation: float = BB_DEVIATION,
    ddof: int = BB_DDOF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(close, period)
    sd = rolling_std(close, period, ddof=ddof)
    upper = mid + deviation * sd
    lower = mid - deviation * sd
    return mid, upper, lower


def shifted(x: np.ndarray, shift: int) -> np.ndarray:
    """MT5 iMA shift: value from `shift` bars earlier. Shift 0 is current."""
    x = np.asarray(x, dtype=np.float64)
    if shift == 0:
        return x.copy()
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if shift > 0:
        out[shift:] = x[:-shift]
    return out
