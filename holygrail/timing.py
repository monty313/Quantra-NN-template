"""live_1m snapshots and developing higher-timeframe bars (spec §2).

Why this file exists: svm-ftmo forward-filled finished 30m/4h values onto every
5m row. That copies the HTF bar's *final* close into snapshots that happened
earlier inside the bar. Spec forbids that.

Instead: at each 1m decision, rebuild the current 5m/30m/1h bar from completed
1m candles from that bar's open through t. Previous F-bars stay frozen.

1m CSV timestamps are bar **open** times.
Decision time t = open + 1 minute (close of that candle).
Current F-bar open = floor(1m open, F). Left-closed, open-time labels.
At t=10:05 the 5m current bar is 10:00; at t=10:06 it is 10:05 with one 1m candle.
"""

from __future__ import annotations

import pandas as pd

from holygrail.config import TF_FREQ


def resample_ohlc(m1: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Completed F-bars from 1m, labeled by open time, left-closed.

    Never includes a 1m candle that opens at the right edge (completes after
    the F-bar's close). Empty bins (weekends) are dropped, matching MT5.
    """
    ohlc = (
        m1[["open", "high", "low", "close"]]
        .resample(freq, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(how="any")
    )
    ohlc.index.name = "bar_open"
    return ohlc


def bar_open_of(open_time: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return open_time.floor(freq)


def developing_ohlc(m1: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Per-1m developing OHLC of the current F-bar (only completed 1m through t)."""
    bar_open = bar_open_of(m1.index, freq)
    g = m1.groupby(bar_open, sort=False)
    out = pd.DataFrame(
        {
            "bar_open": bar_open,
            "open": g["open"].transform("first").to_numpy(),
            "high": g["high"].cummax().to_numpy(),
            "low": g["low"].cummin().to_numpy(),
            "close": m1["close"].to_numpy(),
        },
        index=m1.index,
    )
    out["tp"] = (out["high"] + out["low"] + out["close"]) / 3.0
    return out


def completed_and_dev(m1: pd.DataFrame, tf_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    freq = TF_FREQ[tf_key]
    completed = resample_ohlc(m1, freq)
    completed["tp"] = (completed["high"] + completed["low"] + completed["close"]) / 3.0
    dev = developing_ohlc(m1, freq)
    bar_id = completed.index.get_indexer(dev["bar_open"].to_numpy())
    if (bar_id < 0).any():
        missing = int((bar_id < 0).sum())
        raise RuntimeError(f"{missing} 1m rows have no {tf_key} bar mapping")
    return completed, dev, pd.Series(bar_id, index=m1.index, name="bar_id")
