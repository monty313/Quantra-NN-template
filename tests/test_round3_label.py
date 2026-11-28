"""Round 3 label y_5m_nextclose_vs_typ. Never a feature. y_5m_dir stays."""

from __future__ import annotations

import numpy as np
import pandas as pd

from holygrail.columns import FEATURE_COLUMNS
from holygrail.config import TF_FREQ
from holygrail.features import build_features
from holygrail.labels import build_labels, nextclose_vs_typ
from holygrail.timing import last_row_of_bar, resample_ohlc
from tests.conftest import make_m1


def _this_and_next_5m(m1: pd.DataFrame, i: int):
    t_open = m1.index[i]
    this_bar = t_open.floor("5min")
    next_bar = this_bar + pd.Timedelta(minutes=5)
    return t_open, this_bar, next_bar


def test_nextclose_vs_typ_is_one_or_zero_not_pm():
    m1 = make_m1(800, seed=9)
    y, valid = nextclose_vs_typ(m1)
    assert valid.any()
    vals = y[valid]
    assert np.isin(vals, [0.0, 1.0]).all()
    assert not np.any(vals == -1.0)


def test_nextclose_vs_typ_definition_on_completed_5m():
    m1 = make_m1(800, seed=21)
    y, valid = nextclose_vs_typ(m1)
    last = last_row_of_bar(m1.index, "5min")
    c5 = resample_ohlc(m1, TF_FREQ["5m"])
    tp = (c5["high"] + c5["low"] + c5["close"]) / 3.0
    i = int(np.flatnonzero(last & valid)[12])
    this_bar = m1.index[i].floor("5min")
    next_bar = this_bar + pd.Timedelta(minutes=5)
    expect = 1.0 if float(c5.loc[next_bar, "close"]) > float(tp.loc[this_bar]) else 0.0
    if float(c5.loc[next_bar, "close"]) == float(tp.loc[this_bar]):
        assert not valid[i]
        assert np.isnan(y[i])
    else:
        assert y[i] == expect


def test_nextclose_vs_typ_tie_is_nan_not_class_zero():
    m1 = make_m1(40, start="2024-03-04 00:00:00", seed=1)
    c5 = resample_ohlc(m1, "5min")
    this = c5.index[0]
    nxt = c5.index[1]
    tp_this = (c5.loc[this, "high"] + c5.loc[this, "low"] + c5.loc[this, "close"]) / 3.0
    # Force next 5m close == this typical (tie).
    nxt_mask = m1.index.floor("5min") == nxt
    last_1m = m1.index[nxt_mask][-1]
    m1.loc[last_1m, "close"] = float(tp_this)
    hi = m1.loc[nxt_mask, "high"].to_numpy()
    lo = m1.loc[nxt_mask, "low"].to_numpy()
    m1.loc[last_1m, "high"] = max(float(hi[-1]), float(tp_this))
    m1.loc[last_1m, "low"] = min(float(lo[-1]), float(tp_this))
    y, valid = nextclose_vs_typ(m1)
    i = int(np.flatnonzero(m1.index.floor("5min") == this)[-1])
    assert not valid[i]
    assert np.isnan(y[i])


def test_nextclose_vs_typ_shared_within_5m_bar():
    m1 = make_m1(800, seed=7)
    y, valid = nextclose_vs_typ(m1)
    bar = m1.index.floor(TF_FREQ["5m"])
    frame = pd.DataFrame({"y": y, "v": valid}, index=m1.index)
    for _, g in frame.groupby(bar):
        if not g["v"].any():
            continue
        vals = g.loc[g["v"], "y"].to_numpy()
        assert np.unique(vals).size == 1


def test_nextclose_vs_typ_not_in_x_and_y_5m_dir_kept():
    m1 = make_m1(400, seed=4)
    ft = build_features(m1)
    lab = build_labels(m1, ft)
    assert "y_5m_nextclose_vs_typ" not in ft.X.columns
    assert "y_5m_nextclose_vs_typ" not in FEATURE_COLUMNS
    assert lab.y_5m_dir is lab.y_5m
    assert lab.y_5m_nextclose_vs_typ is not lab.y_5m_dir
    assert lab.label_valid_nextclose_vs_typ.any()


def test_shift_label_one_bar_does_not_change_any_feature_column():
    """Answers live on the next bar. Shifting y by one 5m bar must leave X identical."""
    m1 = make_m1(800, seed=9)
    ft = build_features(m1)
    last = last_row_of_bar(m1.index, "5min")
    X = ft.X.loc[last][FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y, _ = nextclose_vs_typ(m1)
    y_last = y[last]
    y_shift = np.empty_like(y_last)
    y_shift[0] = np.nan
    y_shift[1:] = y_last[:-1]
    # Rebuild from the same candles after the label has been shifted in memory.
    ft2 = build_features(m1)
    X2 = ft2.X.loc[last][FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    assert np.array_equal(np.isnan(X), np.isnan(X2))
    finite = np.isfinite(X)
    assert np.allclose(X[finite], X2[finite], atol=1e-12)
    assert "y_5m_nextclose_vs_typ" not in ft.X.columns
    # Shifted answers are not a feature column.
    for j, name in enumerate(FEATURE_COLUMNS):
        col = X[:, j]
        if np.isfinite(y_shift).sum() < 8:
            continue
        both = np.isfinite(col) & np.isfinite(y_shift)
        if both.sum() < 8:
            continue
        assert not np.array_equal(col[both], y_shift[both]), name


def test_mutating_next_5m_close_changes_y_not_x():
    m1 = make_m1(800, seed=13)
    last = last_row_of_bar(m1.index, "5min")
    y1, v1 = nextclose_vs_typ(m1)
    i = int(np.flatnonzero(last & v1 & (y1 == 1.0))[8])
    t_open, this_bar, next_bar = _this_and_next_5m(m1, i)
    c5 = resample_ohlc(m1, "5min")
    tp_this = float((c5.loc[this_bar, "high"] + c5.loc[this_bar, "low"] + c5.loc[this_bar, "close"]) / 3.0)
    ft = build_features(m1)
    row = ft.X.loc[t_open].to_numpy(dtype=np.float64).copy()
    m2 = m1.copy()
    last_n = m2.index[m2.index.floor("5min") == next_bar][-1]
    new_c = tp_this - 0.05
    m2.loc[last_n, "close"] = new_c
    m2.loc[last_n, "high"] = max(float(m2.loc[last_n, "high"]), new_c, float(m2.loc[last_n, "open"]))
    m2.loc[last_n, "low"] = min(float(m2.loc[last_n, "low"]), new_c, float(m2.loc[last_n, "open"]))
    y2, v2 = nextclose_vs_typ(m2)
    ft2 = build_features(m2)
    row2 = ft2.X.loc[t_open].to_numpy(dtype=np.float64)
    assert np.array_equal(np.isnan(row), np.isnan(row2))
    finite = np.isfinite(row)
    assert np.allclose(row[finite], row2[finite], atol=1e-12, equal_nan=True)
    assert v1[i] and v2[i]
    assert y1[i] == 1.0
    assert y2[i] == 0.0
