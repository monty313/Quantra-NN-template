"""Load MT5 1-minute CSV exports. Timestamps are naive bar-open broker-server times.

Why naive: the file has no timezone column. Assuming UTC would be wrong
(these exports have no Sunday session). Decision_time is open+1m because
MT5 labels the candle by when it *opened*, and the spec's t is when it *closed*.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLC_COLS = ["open", "high", "low", "close"]


def load_mt5_m1(path: str | Path) -> pd.DataFrame:
    """Parse an MT5 History Center CSV (tab-separated <DATE> <TIME> ...).

    Returns a frame indexed by bar **open** time (naive), with columns
    open, high, low, close, tickvol, spread, decision_time.
    decision_time = open + 1 minute (close of that 1m candle).
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]
    if "date" not in df.columns or "time" not in df.columns:
        raise ValueError(f"Expected DATE/TIME columns, got {list(df.columns)}")
    open_time = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
    )
    out = pd.DataFrame(
        {
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
        }
    )
    if "tickvol" in df.columns:
        out["tickvol"] = df["tickvol"].astype("float64")
    if "spread" in df.columns:
        out["spread"] = df["spread"].astype("float64")
    out.index = pd.DatetimeIndex(open_time, name="open_time")
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    if out.index.has_duplicates:
        n = int(out.index.duplicated().sum())
        raise ValueError(f"Duplicate open timestamps: {n}")
    out["decision_time"] = out.index + pd.Timedelta(minutes=1)
    return out


def typical_price(high: pd.Series | pd.DataFrame, low=None, close=None) -> pd.Series:
    if low is None:
        return (high["high"] + high["low"] + high["close"]) / 3.0
    return (high + low + close) / 3.0
