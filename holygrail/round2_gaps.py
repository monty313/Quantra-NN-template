"""Missing-minute flags and 5m gap drops for Round 2.

Mark 2026-11-27:
- Flag incomplete F-bars (1m count short of the clock width). Side flags, not X.
- Drop a 5m row if a *micro-gap* (1 min < δ ≤ 12h) sits in the current 5m bar
  or in the last 12 completed 5m bars (~1h).
- If a hole is longer than 60 minutes (micro or session/weekend), also drop
  the next 199 5m bars after the hole.
- 12h still splits micro vs session. Weekends are session breaks, not micro.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from holygrail.config import (
    GAP_LONG_HOLE_DROP_BARS,
    GAP_LONG_HOLE_MINUTES,
    GAP_SHORT_LOOKBACK_5M,
    MICRO_GAP_MAX_HOURS,
    ROUND2_TFS,
    TF_FREQ,
    TF_MINUTES,
)
from holygrail.timing import last_row_of_bar, resample_ohlc


@dataclass
class GapReport:
    """Side flags. Never columns in X."""

    micro_gap_n: int
    session_break_n: int
    incomplete_completed: dict[str, int]
    n_completed: dict[str, int]
    n_5m_rows: int
    drop_lookback_gap: np.ndarray  # bool, length n_5m_rows (union of rules)
    drop_short: np.ndarray
    drop_long_tail: np.ndarray
    row_incomplete: dict[str, np.ndarray]  # per-TF bool, length n_5m_rows
    last_1m: np.ndarray
    is_last_5m: np.ndarray


def m1_gap_edges(
    index: pd.DatetimeIndex,
    micro_max_hours: float = MICRO_GAP_MAX_HOURS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gap_start, gap_end, is_micro) on the 1m open index.

    gap_start = last present 1m open before the hole.
    gap_end = first present 1m open after the hole.
    is_micro True when 1 min < delta <= micro_max.
    """
    if len(index) < 2:
        empty = np.array([], dtype="datetime64[ns]")
        return empty, empty, np.array([], dtype=bool)
    arr = index.to_numpy()
    delta = np.diff(arr)
    one_min = np.timedelta64(1, "m")
    max_gap = np.timedelta64(int(micro_max_hours * 3600), "s")
    hole = delta > one_min
    is_micro = hole & (delta <= max_gap)
    starts = arr[:-1][hole].astype("datetime64[ns]")
    ends = arr[1:][hole].astype("datetime64[ns]")
    return starts, ends, is_micro[hole]


def completed_minute_counts(m1: pd.DataFrame, tf_key: str) -> pd.Series:
    """How many 1m candles fell in each completed F-bar (open-time label)."""
    freq = TF_FREQ[tf_key]
    bar_open = m1.index.floor(freq)
    counts = pd.Series(1, index=m1.index).groupby(bar_open).sum()
    completed = resample_ohlc(m1, freq)
    return counts.reindex(completed.index).fillna(0).astype(np.int32)


def _bar_minute_counts(m1: pd.DataFrame, tf_key: str) -> pd.Series:
    freq = TF_FREQ[tf_key]
    bar_open = m1.index.floor(freq)
    return pd.Series(1, index=m1.index).groupby(bar_open).sum()


def build_gap_report(
    m1: pd.DataFrame,
    short_lookback_5m: int = GAP_SHORT_LOOKBACK_5M,
    long_hole_minutes: int = GAP_LONG_HOLE_MINUTES,
    long_drop_bars: int = GAP_LONG_HOLE_DROP_BARS,
    micro_max_hours: float = MICRO_GAP_MAX_HOURS,
) -> GapReport:
    index = m1.index
    is_last = last_row_of_bar(index, TF_FREQ["5m"])
    last_1m = index[is_last]
    n_5m = int(is_last.sum())
    last_1m_ns = last_1m.to_numpy(dtype="datetime64[ns]")

    gap_starts, gap_ends, is_micro = m1_gap_edges(index, micro_max_hours)
    micro_starts = gap_starts[is_micro]
    micro_ends = gap_ends[is_micro]

    completed_5m = resample_ohlc(m1, TF_FREQ["5m"])
    c5_idx = completed_5m.index
    bar_open = last_1m.floor(TF_FREQ["5m"])
    bar_id = c5_idx.get_indexer(bar_open)
    bar_close = (bar_open + pd.Timedelta(minutes=5)).to_numpy(dtype="datetime64[ns]")
    start_id = np.clip(bar_id - short_lookback_5m, 0, None)
    lookback_open = np.empty(n_5m, dtype="datetime64[ns]")
    missing = bar_id < 0
    lookback_open[missing] = np.datetime64("NaT", "ns")
    ok = ~missing
    if ok.any():
        lookback_open[ok] = c5_idx.to_numpy()[start_id[ok]]

    drop_short = np.zeros(n_5m, dtype=bool)
    for gs, ge in zip(micro_starts, micro_ends):
        drop_short |= (lookback_open < ge) & (bar_close > gs) & ~np.isnat(lookback_open)

    drop_long = np.zeros(n_5m, dtype=bool)
    long_span = np.timedelta64(long_hole_minutes, "m")
    c5_ns = c5_idx.to_numpy(dtype="datetime64[ns]")
    for gs, ge in zip(gap_starts, gap_ends):
        if (ge - gs) <= long_span:
            continue
        first = int(np.searchsorted(c5_ns, ge, side="left"))
        drop_long |= (bar_id >= first) & (bar_id < first + long_drop_bars) & (bar_id >= 0)

    incomplete_completed: dict[str, int] = {}
    n_completed: dict[str, int] = {}
    row_incomplete: dict[str, np.ndarray] = {}
    for tf in ROUND2_TFS:
        counts_c = completed_minute_counts(m1, tf)
        expected = TF_MINUTES[tf]
        incomplete_completed[tf] = int((counts_c < expected).sum())
        n_completed[tf] = int(len(counts_c))
        live_counts = _bar_minute_counts(m1, tf)
        tf_open = last_1m.floor(TF_FREQ[tf])
        actual = live_counts.reindex(tf_open).fillna(0).to_numpy(dtype=np.int32)
        elapsed = ((last_1m - tf_open) / pd.Timedelta(minutes=1)).to_numpy() + 1.0
        need = np.minimum(elapsed, float(expected))
        row_incomplete[tf] = actual < need

    return GapReport(
        micro_gap_n=int(is_micro.sum()),
        session_break_n=int((~is_micro).sum()),
        incomplete_completed=incomplete_completed,
        n_completed=n_completed,
        n_5m_rows=n_5m,
        drop_lookback_gap=drop_short | drop_long,
        drop_short=drop_short,
        drop_long_tail=drop_long,
        row_incomplete=row_incomplete,
        last_1m=last_1m_ns,
        is_last_5m=is_last,
    )
