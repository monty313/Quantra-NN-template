"""Phase B: blocks 8–11 after 128. y_5m_dir unchanged. No training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from holygrail.columns import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V1_0,
    PHASE_B_COLUMNS,
    SCALER_COLUMNS,
)
from holygrail.config import BB_DDOF, RSI_BB_BB_PERIOD, RSI_BB_DEVIATION, RSI_BB_PERIOD
from holygrail.features import build_features, prev5m_dir_at_decision
from holygrail.indicators import bollinger, rsi_wilder
from holygrail.labels import build_labels
from holygrail.rules import band_state, rbb_label, rsisma_label, s4_state, s4hl_label, sign_np
from holygrail.timing import completed_and_dev, last_row_of_bar
from tests.conftest import make_m1


def test_phase_b_180_columns_first_128_unchanged():
    assert len(FEATURE_COLUMNS_V1_0) == 128
    assert FEATURE_COLUMNS[:128] == FEATURE_COLUMNS_V1_0
    assert FEATURE_COLUMNS[128:] == PHASE_B_COLUMNS
    assert len(FEATURE_COLUMNS) == 180
    assert FEATURE_COLUMNS[128] == "rsisma3_bin_5m"
    assert FEATURE_COLUMNS[146] == "rsisma_label"
    assert FEATURE_COLUMNS[147] == "rbb_pctb_5m"
    assert FEATURE_COLUMNS[159] == "rbb_label"
    assert FEATURE_COLUMNS[160] == "s4_hi_gap_5m"
    assert FEATURE_COLUMNS[178] == "s4hl_label"
    assert FEATURE_COLUMNS[179] == "prev5m_dir"
    m1 = make_m1(400, seed=1)
    ft = build_features(m1)
    assert list(ft.X.columns) == FEATURE_COLUMNS
    assert ft.X.shape[1] == 180


def test_phase_b_scaler_67_gaps_not_prev5m_dir():
    assert len(SCALER_COLUMNS) == 67
    assert "prev5m_dir" not in SCALER_COLUMNS
    for name in (
        "rsisma3_gap_5m",
        "rsisma14_gap_1h",
        "rbb_mid_gap_5m",
        "rbb_mid_gap_1h",
        "s4_hi_gap_5m",
        "s4_lo_gap_30m",
        "s20_gap_1h",
    ):
        assert name in SCALER_COLUMNS
    for raw in (
        "rsisma3_bin_5m",
        "rsisma_label",
        "rbb_pctb_5m",
        "rbb_mid_bin_5m",
        "rbb_band_state_5m",
        "rbb_label",
        "s4_state_5m",
        "s4rbb_pctb_5m",
        "s4rbb_band_state_5m",
        "s4hl_label",
        "prev5m_dir",
    ):
        assert raw not in SCALER_COLUMNS


def test_phase_b_y_not_in_x(m1_short):
    ft = build_features(m1_short)
    for col in (
        "y_5m",
        "y_5m_dir",
        "y_5m_nextclose_vs_typ",
        "y_5m_close2_vs_typ",
        "y_5m_close2_vs_close",
        "y_30m",
        "y_1h_structure",
        "y_15m",
        "y_1h",
    ):
        assert col not in ft.X.columns


def test_phase_b_future_1m_does_not_change_row():
    m1 = make_m1(800, seed=4)
    t_open = m1.index[400]
    ft = build_features(m1)
    row = ft.X.loc[t_open].to_numpy(dtype=np.float64).copy()
    m2 = m1.copy()
    m2.loc[m2.index > t_open, ["open", "high", "low", "close"]] *= 1.5
    ft2 = build_features(m2)
    row2 = ft2.X.loc[t_open].to_numpy(dtype=np.float64)
    assert np.array_equal(np.isnan(row), np.isnan(row2))
    finite = np.isfinite(row)
    assert np.allclose(row[finite], row2[finite], atol=1e-12, equal_nan=True)


def test_prev5m_dir_completed_only_ignores_developing_remainder():
    m1 = make_m1(800, seed=11)
    # First 1m of a 5m bar: k is the previous completed 5m, not the open bar.
    bar_open = m1.index.floor("5min")
    first_of_bar = ~pd.Index(bar_open).duplicated(keep="first")
    i = int(np.where(first_of_bar)[0][20])
    t_open = m1.index[i]
    assert t_open == bar_open[i]
    d0 = prev5m_dir_at_decision(m1)[i]
    m2 = m1.copy()
    later_same_bar = (m2.index > t_open) & (m2.index.floor("5min") == bar_open[i])
    m2.loc[later_same_bar, ["open", "high", "low", "close"]] *= 1.25
    d1 = prev5m_dir_at_decision(m2)[i]
    assert np.isfinite(d0)
    assert d0 == d1
    ft1 = build_features(m1)
    ft2 = build_features(m2)
    assert ft1.X.loc[t_open, "prev5m_dir"] == ft2.X.loc[t_open, "prev5m_dir"]


def test_prev5m_dir_last_1m_uses_this_completed_not_y():
    m1 = make_m1(2000, seed=12)
    completed, _, bar_id = completed_and_dev(m1, "5m")
    tp = ((completed["high"] + completed["low"] + completed["close"]) / 3.0).to_numpy()
    last = last_row_of_bar(m1.index, "5min")
    i = int(np.where(last)[0][40])
    k = int(bar_id.to_numpy()[i])
    expect = float(sign_np(np.array([tp[k] - tp[k - 1]]))[0])
    got = float(prev5m_dir_at_decision(m1)[i])
    assert got == expect
    ft = build_features(m1)
    lab = build_labels(m1, ft)
    assert ft.X.iloc[i]["prev5m_dir"] == expect
    # y is next vs this; prev5m_dir is this vs previous. Not the same definition.
    y = lab.y_5m_dir[i]
    assert np.isfinite(y)
    assert y != 0.0


def test_prev5m_dir_next_bar_mutate_does_not_change_x():
    m1 = make_m1(800, seed=13)
    last = last_row_of_bar(m1.index, "5min")
    i = int(np.where(last)[0][15])
    t_open = m1.index[i]
    this_bar = m1.index.floor("5min")[i]
    next_bar = this_bar + pd.Timedelta(minutes=5)
    ft = build_features(m1)
    row = ft.X.loc[t_open].to_numpy(dtype=np.float64).copy()
    m2 = m1.copy()
    nxt = m2.index.floor("5min") == next_bar
    m2.loc[nxt, ["high", "low", "close"]] *= 0.5
    ft2 = build_features(m2)
    row2 = ft2.X.loc[t_open].to_numpy(dtype=np.float64)
    assert np.array_equal(np.isnan(row), np.isnan(row2))
    finite = np.isfinite(row)
    assert np.allclose(row[finite], row2[finite], atol=1e-12, equal_nan=True)


def test_rsisma_label_formula():
    b3 = np.array([1, 1, 1, -1, -1, 1, 0])
    b7 = np.array([1, 1, -1, -1, -1, 1, 0])
    b14 = np.array([1, -1, -1, -1, 1, -1, 0])
    # 5m two-up, 30m two-up → +1 even if 1h is mixed
    lab = rsisma_label(b3, b7, b14, b3, b7, b14, np.zeros(7, dtype=np.int8), np.zeros(7, dtype=np.int8), np.zeros(7, dtype=np.int8))
    assert lab.tolist() == [1, 1, -1, -1, -1, 1, 0]
    # 5m two-up but neither HTF two-up → 0
    z = np.zeros(7, dtype=np.int8)
    lab2 = rsisma_label(b3, b7, b14, z, z, z, z, z, z)
    assert lab2.tolist() == [0, 0, 0, 0, 0, 0, 0]


def test_rbb_and_s4hl_label_formulas():
    mid30 = np.array([1, 1, -1, -1, 1, 0])
    mid1h = np.array([1, 1, -1, -1, -1, 1])
    band5 = np.array([-1, 1, 1, -1, -1, -1])
    assert rbb_label(mid30, mid1h, band5).tolist() == [1, 0, -1, 0, 0, 0]

    s4_30 = np.array([1, 1, -1, -1, 1])
    s4_1h = np.array([1, 1, -1, -1, 1])
    s20 = np.array([0.1, 0.1, -0.2, -0.2, 0.0])
    band = np.array([-1, 1, 1, -1, -1])
    assert s4hl_label(s4_30, s4_1h, s20, band).tolist() == [1, 0, -1, 0, 0]


def test_s4_state_and_band_state_equality_zero():
    close = np.array([3.0, 0.0, 1.5, 2.0])
    hi = np.array([2.0, 1.0, 2.0, 2.0])
    lo = np.array([1.0, 2.0, 1.0, 2.0])
    assert s4_state(close, hi, lo).tolist() == [1, -1, 0, 0]
    x = np.array([3.0, 0.0, 1.0, 2.0])
    up = np.array([2.0, 2.0, 1.0, 2.0])
    lo_b = np.array([1.0, 1.0, 1.0, 0.0])
    assert band_state(x, up, lo_b).tolist() == [1, -1, 0, 0]


def test_rsi_bb_uses_ddof0_like_mt5_ibands():
    assert BB_DDOF == 0
    assert RSI_BB_PERIOD == 4
    assert RSI_BB_BB_PERIOD == 20
    assert RSI_BB_DEVIATION == 1.0
    x = np.linspace(30.0, 70.0, 25)
    mid, up, lo = bollinger(x, 20, 1.0, ddof=0)
    w = x[-20:]
    m = w.mean()
    sd0 = np.sqrt(np.mean((w - m) ** 2))
    sd1 = np.sqrt(np.sum((w - m) ** 2) / 19)
    assert abs(mid[-1] - m) < 1e-12
    assert abs(up[-1] - (m + sd0)) < 1e-12
    assert abs(lo[-1] - (m - sd0)) < 1e-12
    assert abs(up[-1] - (m + sd1)) > 1e-6


def test_phase_b_labels_match_built_x(m1_long):
    ft = build_features(m1_long)
    X = ft.X.loc[ft.valid]
    if len(X) == 0:
        raise AssertionError("no valid rows")
    rec = rsisma_label(
        X["rsisma3_bin_5m"].to_numpy(),
        X["rsisma7_bin_5m"].to_numpy(),
        X["rsisma14_bin_5m"].to_numpy(),
        X["rsisma3_bin_30m"].to_numpy(),
        X["rsisma7_bin_30m"].to_numpy(),
        X["rsisma14_bin_30m"].to_numpy(),
        X["rsisma3_bin_1h"].to_numpy(),
        X["rsisma7_bin_1h"].to_numpy(),
        X["rsisma14_bin_1h"].to_numpy(),
    )
    assert np.array_equal(X["rsisma_label"].to_numpy(), rec)
    rec_r = rbb_label(
        X["rbb_mid_bin_30m"].to_numpy(),
        X["rbb_mid_bin_1h"].to_numpy(),
        X["rbb_band_state_5m"].to_numpy(),
    )
    assert np.array_equal(X["rbb_label"].to_numpy(), rec_r)
    rec_s = s4hl_label(
        X["s4_state_30m"].to_numpy(),
        X["s4_state_1h"].to_numpy(),
        X["s20_gap_5m"].to_numpy(),
        X["s4rbb_band_state_5m"].to_numpy(),
    )
    assert np.array_equal(X["s4hl_label"].to_numpy(), rec_s)
    assert np.allclose(X["rbb_pctb_5m"].to_numpy(), X["s4rbb_pctb_5m"].to_numpy())
    assert np.array_equal(X["rbb_band_state_5m"].to_numpy(), X["s4rbb_band_state_5m"].to_numpy())


def test_rsisma_on_close_not_typical():
    m1 = make_m1(600, seed=8)
    completed, _, bar_id = completed_and_dev(m1, "5m")
    last = ~pd.Index(bar_id).duplicated(keep="last")
    i = int(np.where(last)[0][-3])
    close_c = completed["close"].to_numpy()
    rsi_c, _, _ = rsi_wilder(close_c, 3)
    k = int(bar_id.to_numpy()[i])
    ft = build_features(m1)
    # last 1m as-of RSI-on-Close equals closed iRSI
    sma5 = pd.Series(rsi_c).rolling(5, min_periods=5).mean().to_numpy()
    gap = rsi_c[k] - sma5[k]
    got = float(ft.X.iloc[i]["rsisma3_gap_5m"])
    assert np.isfinite(got)
    assert abs(got - gap) < 1e-6
    assert float(ft.X.iloc[i]["rsisma3_bin_5m"]) == float(sign_np(np.array([gap]))[0])
