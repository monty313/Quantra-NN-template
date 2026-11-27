"""Time-ordered 70/30 split with purge and 60-minute embargo (spec §13).

Why not sklearn TimeSeriesSplit alone (svm-ftmo): consecutive 1m rows share
the same next-closed HTF label (and used to share a 15m/1h tick window).
Without purge, a train row's label sees test-period prices. Without embargo,
the first test row is 1 minute after the last train row and is not independent.

Why not class_weight='balanced' (adaptive-risk): the spec says report class
counts and do not rebalance. Balancing makes logloss/calibration incomparable
to the real occupancy of the labels.

Shuffle is forbidden. Val is a walk-forward tail inside train, same rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from holygrail.config import EMBARGO_MINUTES, TRAIN_FRACTION, VAL_FRACTION_OF_TRAIN


@dataclass
class Split:
    cutoff: pd.Timestamp  # last training-eligible decision time before embargo
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    inner_cutoff: pd.Timestamp


def time_split(
    decision_time: pd.DatetimeIndex,
    feature_valid: np.ndarray,
    label_valid: np.ndarray,
    label_horizon_minutes: int,
    train_fraction: float = TRAIN_FRACTION,
    embargo_minutes: int = EMBARGO_MINUTES,
    val_fraction_of_train: float = VAL_FRACTION_OF_TRAIN,
) -> Split:
    """Oldest train_fraction of 1m history by row order (time-sorted).

    Purge: drop train rows whose label window reaches past the cutoff
    (for the 1h label, last 60 minutes before cutoff).
    Embargo: first test row at least `embargo_minutes` after the cutoff.
    Validation is the time-ordered tail of the train span, with the same
    purge/embargo against an inner cutoff. Never shuffled.
    """
    n = len(decision_time)
    cutoff_i = int(n * train_fraction) - 1
    cutoff_i = max(0, min(cutoff_i, n - 1))
    cutoff = pd.Timestamp(decision_time[cutoff_i])

    horizon = pd.Timedelta(minutes=label_horizon_minutes)
    embargo = pd.Timedelta(minutes=embargo_minutes)

    usable = feature_valid & label_valid
    dt = pd.DatetimeIndex(decision_time)

    train_end = cutoff - horizon  # last train decision whose window ends at cutoff
    test_start = cutoff + embargo

    train_mask = usable & (dt <= train_end)
    test_mask = usable & (dt >= test_start)

    # Walk-forward val inside the train span
    train_times = dt[train_mask]
    if len(train_times) == 0:
        raise RuntimeError("no training rows after purge")
    inner_cut_pos = int(len(train_times) * (1.0 - val_fraction_of_train)) - 1
    inner_cut_pos = max(0, inner_cut_pos)
    inner_cutoff = pd.Timestamp(train_times[inner_cut_pos])
    fit_end = inner_cutoff - horizon
    val_start = inner_cutoff + embargo
    fit_mask = train_mask & (dt <= fit_end)
    val_mask = train_mask & (dt >= val_start)
    # Rows between fit_end and val_start are dropped (purge+embargo hole)

    return Split(
        cutoff=cutoff,
        train_idx=np.flatnonzero(fit_mask),
        val_idx=np.flatnonzero(val_mask),
        test_idx=np.flatnonzero(test_mask),
        inner_cutoff=inner_cutoff,
    )


def class_counts(y: np.ndarray) -> dict[str, int]:
    y = y[np.isfinite(y)]
    return {
        "-1": int(np.sum(y == -1)),
        "0": int(np.sum(y == 0)),
        "+1": int(np.sum(y == 1)),
        "n": int(y.size),
    }
