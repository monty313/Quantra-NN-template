"""Round 2 Step 3: combined matrix shape, prefixes, 30m-once, gap drop. No train."""

from __future__ import annotations

import numpy as np
import pandas as pd

from holygrail.round2_columns import (
    N_PER_TF,
    N_ROUND2_COLUMNS,
    ROUND2_FEATURE_COLUMNS,
)
from holygrail.round2_gaps import build_gap_report, m1_gap_edges
from holygrail.round2_matrix import build_round2_matrix, count_report
from tests.conftest import make_m1


def test_column_count_and_30m_once():
    assert len(ROUND2_FEATURE_COLUMNS) == N_ROUND2_COLUMNS == 310
    assert len(set(ROUND2_FEATURE_COLUMNS)) == 310
    assert sum(c.startswith("30m_") for c in ROUND2_FEATURE_COLUMNS) == N_PER_TF
    assert sum(c.startswith("1m_") for c in ROUND2_FEATURE_COLUMNS) == N_PER_TF
    assert sum(c.startswith("15m_") for c in ROUND2_FEATURE_COLUMNS) == N_PER_TF
    assert "s1_sma_ab_cons" in ROUND2_FEATURE_COLUMNS
    assert "s2_sma_ab_cons" in ROUND2_FEATURE_COLUMNS
    assert "s1_rsi_htf_cons" in ROUND2_FEATURE_COLUMNS
    assert "s2_rsi_htf_cons" in ROUND2_FEATURE_COLUMNS
    assert ROUND2_FEATURE_COLUMNS[-1] == "prev5m_dir"
    # No TOD / session features.
    joined = " ".join(ROUND2_FEATURE_COLUMNS)
    assert "tod" not in joined.lower()
    assert "session" not in joined.lower()
    assert "hour_of" not in joined.lower()


def test_micro_gap_not_weekend():
    idx = pd.DatetimeIndex(
        [
            "2024-03-08 23:57",
            "2024-03-08 23:58",
            "2024-03-11 00:00",  # ~48h weekend
            "2024-03-11 00:01",
            "2024-03-11 00:05",  # 3-minute hole
            "2024-03-11 00:06",
        ]
    )
    starts, ends, is_micro = m1_gap_edges(idx)
    assert is_micro.sum() == 1
    assert (~is_micro).sum() == 1
    assert starts[is_micro][0] == np.datetime64("2024-03-11 00:01")
    assert ends[is_micro][0] == np.datetime64("2024-03-11 00:05")


def test_lookback_drop_on_punched_hole():
    m1 = make_m1(400, seed=4)
    # Punch a 3-minute hole in the middle of a 5m bar.
    hole = m1.index[200:203]
    punched = m1.drop(hole)
    report = build_gap_report(punched, warmup_bars=3)
    assert report.micro_gap_n == 1
    assert int(report.drop_lookback_gap.sum()) > 0
    # Rows whose last 1m is before the hole should not drop (warmup 3 5m bars
    # is 15 minutes; hole is at row 200).
    last = report.last_1m
    hole_start = hole[0].to_datetime64()
    before = last < hole_start
    # First 5m rows well before the hole: lookback cannot reach it if
    # last_1m + warmup window ends before the hole.
    assert not report.drop_lookback_gap[before][:5].any()


def test_build_round2_smoke_columns():
    m1 = make_m1(800, seed=5)
    table = build_round2_matrix(m1)
    assert list(table.X.columns) == ROUND2_FEATURE_COLUMNS
    assert len(table.X) == int(table.gap_report.n_5m_rows)
    assert table.X.shape[1] == 310
    counts = count_report(table, m1)
    assert counts["n_columns"] == 310
    assert counts["n_dropped_lookback_gap"] + counts["n_rows_after_gap_drop"] == counts[
        "n_5m_rows_before_gap_drop"
    ]
    assert counts["y_5m_close_model_acc"] is None
