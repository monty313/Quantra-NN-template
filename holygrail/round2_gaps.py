"""Missing-minute flags and lookback-crosses-gap drops for Round 2.

Weekend/session-close holes (FX ~48h) are not "missing minutes". A 1m delta
in (1 minute, MICRO_GAP_MAX_HOURS] is an unexpected hole. Bars whose 1m
count is short of the broker-clock width are flagged. A 5m row is dropped
when any such hole sits inside the indicator lookback on any TF.

Why 12h: EURUSD weekend is ~48–72h. Intra-session holes in Step 0 were 2–7
minutes. 12h sits between those. If Mark wants weekend-crossing lookbacks
dropped too, say so — that would empty the matrix (1h CCI100 always spans
a weekend).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from holygrail.config import (
    MICRO_GAP_MAX_HOURS,
    ROUND2_TFS,
    TF_FREQ,
    TF_MINUTES,
    WARMUP_BARS,
)
from holygrail.timing import last_row_of_bar, resample_ohlc


@dataclass
class GapReport:
    """Side flags. Never columns in X."""

    micro_gap_n: int
    session_break_n: int
    # Per TF, completed bars with 1m count < TF minutes.
    incomplete_completed: dict[str, int]
    n_completed: dict[str, int]
    # last-1m-of-5m grain
    n_5m_rows: int
    drop_lookback_gap: np.ndarray  # bool, length n_5m_rows
    lookback_start: np.ndarray  # datetime64[ns], length n_5m_rows
    last_1m: np.ndarray
    is_last_5m: np.ndarray  # bool on the 1m index


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
    # pandas 1m indexes are often datetime64[us]; compare timedeltas, not raw ints.
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


def lookback_start_on_tf(
    m1: pd.DataFrame,
    tf_key: str,
    last_1m: pd.DatetimeIndex,
    warmup_bars: int = WARMUP_BARS,
) -> np.ndarray:
    """Open time of the oldest completed F-bar in the CCI100 warmup window.

    Developing current bar is included as bar_id; lookback is bar_id - warmup.
    """
    freq = TF_FREQ[tf_key]
    completed = resample_ohlc(m1, freq)
    current_open = last_1m.floor(freq)
    bar_id = completed.index.get_indexer(current_open)
    start_id = np.clip(bar_id - warmup_bars, 0, None)
    out = np.empty(len(last_1m), dtype="datetime64[ns]")
    missing = bar_id < 0
    out[missing] = np.datetime64("NaT", "ns")
    ok = ~missing
    if ok.any():
        out[ok] = completed.index.to_numpy()[start_id[ok]]
    return out


def build_gap_report(
    m1: pd.DataFrame,
    warmup_bars: int = WARMUP_BARS,
    micro_max_hours: float = MICRO_GAP_MAX_HOURS,
) -> GapReport:
    index = m1.index
    is_last = last_row_of_bar(index, TF_FREQ["5m"])
    last_1m = index[is_last]
    n_5m = int(is_last.sum())

    gap_starts, gap_ends, is_micro = m1_gap_edges(index, micro_max_hours)
    micro_starts = gap_starts[is_micro]
    micro_ends = gap_ends[is_micro]

    incomplete_completed: dict[str, int] = {}
    n_completed: dict[str, int] = {}
    lookbacks = []
    for tf in ROUND2_TFS:
        counts = completed_minute_counts(m1, tf)
        expected = TF_MINUTES[tf]
        incomplete_completed[tf] = int((counts < expected).sum())
        n_completed[tf] = int(len(counts))
        lookbacks.append(lookback_start_on_tf(m1, tf, last_1m, warmup_bars))

    # Longest window: earliest lookback start among TFs (NaT ignored).
    stacked = np.stack(lookbacks, axis=0)
    # datetime64 min: NaT is min in numpy — replace NaT with max so they don't win.
    nat = np.isnat(stacked)
    fill = np.datetime64("NaT", "ns")
    if nat.all():
        lookback_start = np.empty(n_5m, dtype="datetime64[ns]")
        lookback_start[:] = fill
    else:
        stacked = stacked.copy()
        stacked[nat] = np.datetime64("2099-01-01")
        lookback_start = stacked.min(axis=0)
        still_dummy = lookback_start == np.datetime64("2099-01-01")
        lookback_start[still_dummy] = fill

    last_1m_ns = last_1m.to_numpy(dtype="datetime64[ns]")
    drop = np.zeros(n_5m, dtype=bool)
    for gs, ge in zip(micro_starts, micro_ends):
        # Hole (gs, ge) overlaps [lookback_start, last_1m].
        drop |= (lookback_start < ge) & (last_1m_ns > gs) & ~np.isnat(lookback_start)

    return GapReport(
        micro_gap_n=int(is_micro.sum()),
        session_break_n=int((~is_micro).sum()),
        incomplete_completed=incomplete_completed,
        n_completed=n_completed,
        n_5m_rows=n_5m,
        drop_lookback_gap=drop,
        lookback_start=lookback_start,
        last_1m=last_1m_ns,
        is_last_5m=is_last,
    )
