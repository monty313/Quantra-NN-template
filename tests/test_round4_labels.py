"""Round 4 two-bar labels. Never features. Leak tests before any score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from holygrail.columns import FEATURE_COLUMNS
from holygrail.config import TF_FREQ
from holygrail.features import build_features
from holygrail.labels import build_labels, close2_vs_close, close2_vs_typ, nextclose_vs_close
from holygrail.timing import last_row_of_bar, resample_ohlc
from tests.conftest import make_m1


def _bars(m1: pd.DataFrame, i: int):
    t_open = m1.index[i]
    this_bar = t_open.floor("5min")
    return t_open, this_bar, this_bar + pd.Timedelta(minutes=10)


def test_close2_labels_are_one_or_zero():
    m1 = make_m1(800, seed=9)
    for fn in (close2_vs_typ, close2_vs_close):
        y, valid = fn(m1)
        assert valid.any()
        vals = y[valid]
        assert np.isin(vals, [0.0, 1.0]).all()
        assert not np.any(vals == -1.0)


def test_close2_vs_typ_definition():
    m1 = make_m1(800, seed=21)
    y, valid = close2_vs_typ(m1)
    last = last_row_of_bar(m1.index, "5min")
    c5 = resample_ohlc(m1, TF_FREQ["5m"])
    tp = (c5["high"] + c5["low"] + c5["close"]) / 3.0
    i = int(np.flatnonzero(last & valid)[12])
    this_bar = m1.index[i].floor("5min")
    t2 = this_bar + pd.Timedelta(minutes=10)
    expect = 1.0 if float(c5.loc[t2, "close"]) > float(tp.loc[this_bar]) else 0.0
    assert y[i] == expect


def test_close2_vs_close_definition():
    m1 = make_m1(800, seed=21)
    y, valid = close2_vs_close(m1)
    last = last_row_of_bar(m1.index, "5min")
    c5 = resample_ohlc(m1, TF_FREQ["5m"])
    i = int(np.flatnonzero(last & valid)[12])
    this_bar = m1.index[i].floor("5min")
    t2 = this_bar + pd.Timedelta(minutes=10)
    expect = 1.0 if float(c5.loc[t2, "close"]) > float(c5.loc[this_bar, "close"]) else 0.0
    assert y[i] == expect


def test_missing_t2_bar_is_nan_not_zero():
    m1 = make_m1(40, start="2024-03-04 00:00:00", seed=14)
    drop = pd.date_range("2024-03-04 00:10:00", periods=5, freq="min")
    m1 = m1.drop(index=drop)
    y_a, v_a = close2_vs_typ(m1)
    y_b, v_b = close2_vs_close(m1)
    i = 0  # 00:00 bar; t+2 is 00:10, missing
    assert not v_a[i] and not v_b[i]
    assert np.isnan(y_a[i]) and np.isnan(y_b[i])


def test_close2_tie_is_nan_not_class_zero():
    m1 = make_m1(60, start="2024-03-04 00:00:00", seed=1)
    c5 = resample_ohlc(m1, "5min")
    this = c5.index[0]
    t2 = c5.index[2]
    tp_this = float((c5.loc[this, "high"] + c5.loc[this, "low"] + c5.loc[this, "close"]) / 3.0)
    t2_mask = m1.index.floor("5min") == t2
    last_1m = m1.index[t2_mask][-1]
    m1.loc[last_1m, "close"] = tp_this
    m1.loc[last_1m, "high"] = max(float(m1.loc[last_1m, "high"]), tp_this)
    m1.loc[last_1m, "low"] = min(float(m1.loc[last_1m, "low"]), tp_this)
    y, valid = close2_vs_typ(m1)
    i = int(np.flatnonzero(m1.index.floor("5min") == this)[-1])
    assert not valid[i]
    assert np.isnan(y[i])


def test_close2_shared_within_5m_bar():
    m1 = make_m1(800, seed=7)
    y, valid = close2_vs_typ(m1)
    bar = m1.index.floor(TF_FREQ["5m"])
    frame = pd.DataFrame({"y": y, "v": valid}, index=m1.index)
    for _, g in frame.groupby(bar):
        if not g["v"].any():
            continue
        assert np.unique(g.loc[g["v"], "y"].to_numpy()).size == 1


def test_close2_labels_not_in_x():
    m1 = make_m1(400, seed=4)
    ft = build_features(m1)
    lab = build_labels(m1, ft)
    for name in ("y_5m_close2_vs_typ", "y_5m_close2_vs_close"):
        assert name not in ft.X.columns
        assert name not in FEATURE_COLUMNS
    assert lab.label_valid_close2_vs_typ.any()
    assert lab.label_valid_close2_vs_close.any()
    assert lab.y_5m_dir is lab.y_5m


def test_shift_y_one_bar_does_not_change_X():
    m1 = make_m1(800, seed=9)
    ft = build_features(m1)
    last = last_row_of_bar(m1.index, "5min")
    X = ft.X.loc[last][FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    ft2 = build_features(m1)
    X2 = ft2.X.loc[last][FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    assert np.array_equal(np.isnan(X), np.isnan(X2))
    finite = np.isfinite(X)
    assert np.allclose(X[finite], X2[finite], atol=1e-12)
    for fn in (close2_vs_typ, close2_vs_close):
        y, _ = fn(m1)
        y_last = y[last]
        y_shift = np.empty_like(y_last)
        y_shift[0] = np.nan
        y_shift[1:] = y_last[:-1]
        for j, name in enumerate(FEATURE_COLUMNS):
            col = X[:, j]
            both = np.isfinite(col) & np.isfinite(y_shift)
            if both.sum() < 8:
                continue
            assert not np.array_equal(col[both], y_shift[both]), name


def test_mutate_t2_close_flips_A_not_X():
    m1 = make_m1(800, seed=13)
    last = last_row_of_bar(m1.index, "5min")
    y1, v1 = close2_vs_typ(m1)
    i = int(np.flatnonzero(last & v1 & (y1 == 1.0))[8])
    t_open, this_bar, t2_bar = _bars(m1, i)
    c5 = resample_ohlc(m1, "5min")
    tp_this = float((c5.loc[this_bar, "high"] + c5.loc[this_bar, "low"] + c5.loc[this_bar, "close"]) / 3.0)
    ft = build_features(m1)
    row = ft.X.loc[t_open].to_numpy(dtype=np.float64).copy()
    m2 = m1.copy()
    last_n = m2.index[m2.index.floor("5min") == t2_bar][-1]
    new_c = tp_this - 0.05
    m2.loc[last_n, "close"] = new_c
    m2.loc[last_n, "high"] = max(float(m2.loc[last_n, "high"]), new_c, float(m2.loc[last_n, "open"]))
    m2.loc[last_n, "low"] = min(float(m2.loc[last_n, "low"]), new_c, float(m2.loc[last_n, "open"]))
    y2, v2 = close2_vs_typ(m2)
    row2 = build_features(m2).X.loc[t_open].to_numpy(dtype=np.float64)
    assert np.array_equal(np.isnan(row), np.isnan(row2))
    finite = np.isfinite(row)
    assert np.allclose(row[finite], row2[finite], atol=1e-12, equal_nan=True)
    assert v1[i] and v2[i]
    assert y1[i] == 1.0
    assert y2[i] == 0.0


def test_mutate_t2_close_flips_B_not_X():
    m1 = make_m1(800, seed=17)
    last = last_row_of_bar(m1.index, "5min")
    y1, v1 = close2_vs_close(m1)
    i = int(np.flatnonzero(last & v1 & (y1 == 1.0))[8])
    t_open, this_bar, t2_bar = _bars(m1, i)
    c5 = resample_ohlc(m1, "5min")
    close_this = float(c5.loc[this_bar, "close"])
    ft = build_features(m1)
    row = ft.X.loc[t_open].to_numpy(dtype=np.float64).copy()
    m2 = m1.copy()
    last_n = m2.index[m2.index.floor("5min") == t2_bar][-1]
    new_c = close_this - 0.05
    m2.loc[last_n, "close"] = new_c
    m2.loc[last_n, "high"] = max(float(m2.loc[last_n, "high"]), new_c, float(m2.loc[last_n, "open"]))
    m2.loc[last_n, "low"] = min(float(m2.loc[last_n, "low"]), new_c, float(m2.loc[last_n, "open"]))
    y2, v2 = close2_vs_close(m2)
    row2 = build_features(m2).X.loc[t_open].to_numpy(dtype=np.float64)
    assert np.array_equal(np.isnan(row), np.isnan(row2))
    finite = np.isfinite(row)
    assert np.allclose(row[finite], row2[finite], atol=1e-12, equal_nan=True)
    assert v1[i] and v2[i]
    assert y1[i] == 1.0
    assert y2[i] == 0.0


def test_nextclose_vs_close_definition_and_not_in_x():
    m1 = make_m1(800, seed=21)
    y, valid = nextclose_vs_close(m1)
    last = last_row_of_bar(m1.index, "5min")
    c5 = resample_ohlc(m1, TF_FREQ["5m"])
    i = int(np.flatnonzero(last & valid)[12])
    this_bar = m1.index[i].floor("5min")
    nxt = this_bar + pd.Timedelta(minutes=5)
    expect = 1.0 if float(c5.loc[nxt, "close"]) > float(c5.loc[this_bar, "close"]) else 0.0
    assert y[i] == expect
    ft = build_features(m1)
    assert "y_5m_nextclose_vs_close" not in ft.X.columns
    assert "y_5m_nextclose_vs_close" not in FEATURE_COLUMNS
