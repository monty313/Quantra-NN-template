"""Structure-state labels: next closed 5m/30m/1h, shared per F-bar, no leak."""

from __future__ import annotations

import numpy as np
import pandas as pd

from holygrail.config import STRUCTURE_ATR_FRACTION, TF_FREQ
from holygrail.features import build_features
from holygrail.labels import build_labels, structure_direction
from holygrail.timing import last_row_of_bar
from tests.conftest import make_m1


def test_structure_shared_within_5m_bar():
    m1 = make_m1(800, seed=7)
    y, valid, _ = structure_direction(m1, "5m")
    bar = m1.index.floor(TF_FREQ["5m"])
    frame = pd.DataFrame({"y": y, "v": valid}, index=m1.index)
    for _, g in frame.groupby(bar):
        if not g["v"].any():
            continue
        vals = g.loc[g["v"], "y"].to_numpy()
        assert np.unique(vals).size == 1


def test_structure_missing_next_bar_is_nan_not_zero():
    m1 = make_m1(40, start="2024-03-04 00:00:00", seed=14)
    # Drop the entire next 5m bar after 00:00 (opens 00:05..00:09)
    drop = pd.date_range("2024-03-04 00:05:00", periods=5, freq="min")
    m1 = m1.drop(index=drop)
    y, valid, _ = structure_direction(m1, "5m")
    # Row at 00:00: current 5m=00:00, next closed after current should be 00:05 — missing
    assert not valid[0]
    assert np.isnan(y[0])


def test_structure_no_leak_after_next_bar_closes():
    m1 = make_m1(400, seed=31)
    y1, v1, _ = structure_direction(m1, "5m")
    i = 20  # 00:20 → current 5m 00:20, next closed 00:25 completes at 00:30
    t_open = m1.index[i]
    current = t_open.floor("5min")
    next_close = current + pd.Timedelta(minutes=10)  # next bar open+5m
    m2 = m1.copy()
    m2.loc[m2.index >= next_close, ["high", "low", "close"]] *= 1.25
    y2, v2, _ = structure_direction(m2, "5m")
    if not v1[i]:
        assert not v2[i]
        assert np.isnan(y2[i])
    else:
        assert y1[i] == y2[i]


def test_structure_uses_same_htf_atr_fraction():
    m1 = make_m1(800, seed=3)
    y, valid, th = structure_direction(m1, "5m", atr_fraction=STRUCTURE_ATR_FRACTION)
    assert STRUCTURE_ATR_FRACTION == 0.5
    assert valid.any()
    assert np.isfinite(th[valid]).all()
    assert (th[valid] > 0).all()
    # threshold is 0.5× ATR of last completed 5m, so it is constant inside a 5m bar
    bar = m1.index.floor("5min")
    frame = pd.DataFrame({"th": th, "v": valid}, index=m1.index)
    for _, g in frame.groupby(bar):
        if not g["v"].any():
            continue
        ths = g.loc[g["v"], "th"].to_numpy()
        assert np.allclose(ths, ths[0])


def test_structure_shared_within_1h_bar():
    m1 = make_m1(2000, seed=11)
    y, valid, _ = structure_direction(m1, "1h")
    bar = m1.index.floor(TF_FREQ["1h"])
    frame = pd.DataFrame({"y": y, "v": valid}, index=m1.index)
    n_shared = 0
    for _, g in frame.groupby(bar):
        if not g["v"].any():
            continue
        vals = g.loc[g["v"], "y"].to_numpy()
        assert np.unique(vals).size == 1
        n_shared += 1
    assert n_shared > 0


def test_structure_1h_missing_next_bar_is_nan_not_zero():
    m1 = make_m1(150, start="2024-03-04 00:00:00", seed=8)
    drop = pd.date_range("2024-03-04 01:00:00", periods=60, freq="min")
    m1 = m1.drop(index=drop)
    y, valid, _ = structure_direction(m1, "1h")
    assert not valid[0]
    assert np.isnan(y[0])


def test_structure_30m_invalid_when_next_30m_missing():
    m1 = make_m1(80, start="2024-03-04 00:00:00", seed=5)
    drop = pd.date_range("2024-03-04 00:30:00", periods=30, freq="min")
    m1 = m1.drop(index=drop)
    y, valid, _ = structure_direction(m1, "30m")
    assert not valid[0]
    assert np.isnan(y[0])


def test_build_labels_keeps_old_heads_and_adds_structure():
    m1 = make_m1(2000, seed=9)
    ft = build_features(m1)
    lab = build_labels(m1, ft)
    assert lab.y_5m.shape == lab.y_15m.shape == (len(m1),)
    assert lab.label_valid_5m.any() and lab.label_valid_30m.any()
    assert lab.y_1h_structure.shape == (len(m1),)
    assert lab.label_valid_1h_structure.any()
    # old comparison heads still exist
    assert lab.label_valid_15m.any()
    last = len(m1) - 1
    assert not lab.label_valid_15m[last]
    assert not lab.label_valid_5m[last]


def test_last_row_of_bar_one_per_5m():
    m1 = make_m1(20, seed=0)
    last = last_row_of_bar(m1.index, "5min")
    assert last.sum() == 4  # 20 minutes → 4 complete 5m bars
    assert last[4] and last[9] and last[14] and last[19]
