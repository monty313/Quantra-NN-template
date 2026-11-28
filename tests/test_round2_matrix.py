"""Round 2: combined matrix, Mark's gap rule, 310 cols. No train here."""

from __future__ import annotations

import numpy as np
import pandas as pd

from holygrail.round2_columns import (
    N_PER_TF,
    N_ROUND2_COLUMNS,
    ROUND2_FEATURE_COLUMNS,
    ROUND2_SCALER_COLUMNS,
)
from holygrail.round2_gaps import build_gap_report, m1_gap_edges
from holygrail.round2_matrix import build_round2_matrix, count_report
from tests.conftest import make_m1


def test_column_count_and_30m_once():
    assert len(ROUND2_FEATURE_COLUMNS) == N_ROUND2_COLUMNS == 310
    assert len(set(ROUND2_FEATURE_COLUMNS)) == 310
    assert sum(c.startswith("30m_") for c in ROUND2_FEATURE_COLUMNS) == N_PER_TF
    assert "s1_sma_ab_cons" in ROUND2_FEATURE_COLUMNS
    assert "s2_sma_ab_cons" in ROUND2_FEATURE_COLUMNS
    assert ROUND2_FEATURE_COLUMNS[-1] == "prev5m_dir"
    joined = " ".join(ROUND2_FEATURE_COLUMNS)
    assert "tod" not in joined.lower()
    assert "session" not in joined.lower()
    assert len(ROUND2_SCALER_COLUMNS) == 115
    assert "prev5m_dir" not in ROUND2_SCALER_COLUMNS


def test_micro_gap_not_weekend():
    idx = pd.DatetimeIndex(
        [
            "2024-03-08 23:57",
            "2024-03-08 23:58",
            "2024-03-11 00:00",
            "2024-03-11 00:01",
            "2024-03-11 00:05",
            "2024-03-11 00:06",
        ]
    )
    starts, ends, is_micro = m1_gap_edges(idx)
    assert is_micro.sum() == 1
    assert (~is_micro).sum() == 1
    assert starts[is_micro][0] == np.datetime64("2024-03-11 00:01")
    assert ends[is_micro][0] == np.datetime64("2024-03-11 00:05")


def test_short_lookback_drops_current_and_next_12():
    m1 = make_m1(400, seed=4)
    hole = m1.index[200:203]
    punched = m1.drop(hole)
    report = build_gap_report(punched)
    assert report.micro_gap_n == 1
    n_drop = int(report.drop_lookback_gap.sum())
    # Current 5m + last 12 completed that still see the hole ≈ ≤ 13 5m rows
    # (plus the 5m that contains the hole on either side). Not the old 1h-CCI tail.
    assert 1 <= n_drop <= 20
    last = report.last_1m
    hole_start = hole[0].to_datetime64()
    before = last < np.datetime64("2024-03-04 03:00")
    assert not report.drop_lookback_gap[before].any()


def test_long_hole_drops_next_199_5m():
    """A 2-hour hole (>60m) drops the next 199 5m bars after it."""
    left = make_m1(300, start="2024-03-04 00:00:00", seed=1)
    right = make_m1(2000, start="2024-03-04 07:00:00", seed=2)
    m1 = pd.concat([left, right])
    report = build_gap_report(m1)
    assert report.micro_gap_n == 1
    # 199 5m after the hole, plus short-lookback rows that still see it.
    n_long = int(report.drop_long_tail.sum())
    assert n_long == 199
    n_short = int(report.drop_short.sum())
    assert n_short > 0
    assert int(report.drop_lookback_gap.sum()) >= 199


def test_build_round2_smoke_columns():
    m1 = make_m1(800, seed=5)
    table = build_round2_matrix(m1)
    assert list(table.X.columns) == ROUND2_FEATURE_COLUMNS
    assert table.X.shape[1] == 310
    counts = count_report(table, m1)
    assert counts["n_columns"] == 310
    assert counts["n_dropped_lookback_gap"] + counts["n_rows_after_gap_drop"] == counts[
        "n_5m_rows_before_gap_drop"
    ]
