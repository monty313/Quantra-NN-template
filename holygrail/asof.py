"""As-of (developing last bar) indicator updates.

Previous F-bars are frozen at their completed values. The current F-bar is
rebuilt from 1m candles through decision time t. At an exact bar boundary the
current bar is the one that just completed, so as-of matches closed-bar values
(that identity is how we test T12/T16 without an MT5 terminal).

idx is the integer position of the current F-bar in the completed series.
Lookbacks (c-2, SMA shift 1/2/3) always read completed[idx - k], never a
developing value — "previous bars are final."
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from holygrail.config import ATR_PERIOD, CCI_CONSTANT, STD_DDOF
from holygrail.indicators import atr_wilder, rolling_sum, rsi_wilder, true_range


def asof_sma(completed: np.ndarray, idx: np.ndarray, current: np.ndarray, period: int) -> np.ndarray:
    completed = np.asarray(completed, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    current = np.asarray(current, dtype=np.float64)
    n = idx.size
    out = np.full(n, np.nan, dtype=np.float64)
    if period == 1:
        return current.copy()
    rs = rolling_sum(completed, period - 1)
    ok = idx >= (period - 1)
    out[ok] = (rs[idx[ok] - 1] + current[ok]) / period
    return out


def asof_std(
    completed: np.ndarray,
    idx: np.ndarray,
    current: np.ndarray,
    period: int,
    ddof: int = STD_DDOF,
) -> np.ndarray:
    completed = np.asarray(completed, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    current = np.asarray(current, dtype=np.float64)
    n = idx.size
    out = np.full(n, np.nan, dtype=np.float64)
    w = period - 1
    if w <= 0:
        return out
    rs = rolling_sum(completed, w)
    rs2 = rolling_sum(completed * completed, w)
    ok = idx >= w
    s = rs[idx[ok] - 1] + current[ok]
    s2 = rs2[idx[ok] - 1] + current[ok] * current[ok]
    sse = s2 - (s * s) / period
    denom = period - ddof
    out[ok] = np.sqrt(np.maximum(sse / denom, 0.0))
    return out


def asof_cci(
    completed_tp: np.ndarray,
    idx: np.ndarray,
    current_tp: np.ndarray,
    period: int,
    constant: float = CCI_CONSTANT,
    chunk: int = 250_000,
) -> np.ndarray:
    completed_tp = np.asarray(completed_tp, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    current_tp = np.asarray(current_tp, dtype=np.float64)
    n = idx.size
    out = np.full(n, np.nan, dtype=np.float64)
    w = period - 1
    if completed_tp.size < w:
        return out
    windows = sliding_window_view(completed_tp, w)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        ii = idx[sl]
        cur = current_tp[sl]
        ok = ii >= w
        if not np.any(ok):
            continue
        prev = np.asarray(windows[ii[ok] - w], dtype=np.float64)
        cur_ok = cur[ok]
        mean = (prev.sum(axis=1) + cur_ok) / period
        mad = (np.abs(prev - mean[:, None]).sum(axis=1) + np.abs(cur_ok - mean)) / period
        val = np.full(mean.shape, np.nan)
        np.divide(cur_ok - mean, constant * mad, out=val, where=mad > 0)
        tmp = out[sl]
        tmp[ok] = val
        out[sl] = tmp
    return out


def asof_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    idx: np.ndarray,
    dh: np.ndarray,
    dl: np.ndarray,
    period: int = ATR_PERIOD,
) -> np.ndarray:
    """Wilder ATR with developing last bar."""
    tr = true_range(high, low, close)
    atr_c = atr_wilder(high, low, close, period)
    idx = np.asarray(idx, dtype=np.int64)
    dh = np.asarray(dh, dtype=np.float64)
    dl = np.asarray(dl, dtype=np.float64)
    n = idx.size
    prev_c = np.full(n, np.nan, dtype=np.float64)
    ok_prev = idx > 0
    prev_c[ok_prev] = close[idx[ok_prev] - 1]
    tr_dev = np.maximum(dh - dl, np.maximum(np.abs(dh - prev_c), np.abs(dl - prev_c)))
    out = np.full(n, np.nan, dtype=np.float64)
    seeded = idx >= period
    out[seeded] = (atr_c[idx[seeded] - 1] * (period - 1) + tr_dev[seeded]) / period
    seed = idx == (period - 1)
    if np.any(seed):
        s0 = tr[: period - 1].sum()
        out[seed] = (s0 + tr_dev[seed]) / period
    return out


def asof_rsi(
    price_c: np.ndarray,
    idx: np.ndarray,
    price_dev: np.ndarray,
    period: int,
) -> np.ndarray:
    rsi_c, ag_c, al_c = rsi_wilder(price_c, period)
    idx = np.asarray(idx, dtype=np.int64)
    price_dev = np.asarray(price_dev, dtype=np.float64)
    n = idx.size
    prev = np.full(n, np.nan, dtype=np.float64)
    ok_prev = idx > 0
    prev[ok_prev] = price_c[idx[ok_prev] - 1]
    chg = price_dev - prev
    gain = np.clip(chg, 0.0, None)
    loss = np.clip(-chg, 0.0, None)
    out = np.full(n, np.nan, dtype=np.float64)
    # After seed: idx-1 >= period so idx >= period+1
    ok = idx >= (period + 1)
    ag = (ag_c[idx[ok] - 1] * (period - 1) + gain[ok]) / period
    al = (al_c[idx[ok] - 1] * (period - 1) + loss[ok]) / period
    out[ok] = _rsi_from_avg(ag, al)
    # Seed bar: idx == period, SMA of first period-1 completed changes + developing change
    seed = idx == period
    if np.any(seed):
        d = np.diff(price_c)
        g0 = np.clip(d[: period - 1], 0.0, None).sum()
        l0 = np.clip(-d[: period - 1], 0.0, None).sum()
        ags = (g0 + gain[seed]) / period
        als = (l0 + loss[seed]) / period
        out[seed] = _rsi_from_avg(ags, als)
    return out


def _rsi_from_avg(ag: np.ndarray, al: np.ndarray) -> np.ndarray:
    both0 = (ag == 0) & (al == 0)
    al0 = (al == 0) & (ag > 0)
    ag0 = (ag == 0) & (al > 0)
    rs = np.divide(ag, al, out=np.full_like(ag, np.nan), where=al > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = np.where(both0, 50.0, rsi)
    rsi = np.where(al0, 100.0, rsi)
    rsi = np.where(ag0, 0.0, rsi)
    return rsi


def lookback(completed: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """Value of a completed-bar series k bars before the current bar (frozen)."""
    idx = np.asarray(idx, dtype=np.int64)
    out = np.full(idx.shape, np.nan, dtype=np.float64)
    ok = idx >= k
    out[ok] = completed[idx[ok] - k]
    return out
