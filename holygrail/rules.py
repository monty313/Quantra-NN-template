"""Pure sign / consensus / state rules. Exact equality → 0. No tolerance."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def sign_np(x: Array) -> Array:
    """+1 / -1 / 0 with exact equality → 0."""
    out = np.zeros(np.shape(x), dtype=np.int8)
    out = np.where(x > 0, 1, out)
    out = np.where(x < 0, -1, out)
    return out.astype(np.int8)


def cci_level(cci: Array, zone: float = 50.0) -> Array:
    out = np.zeros(np.shape(cci), dtype=np.int8)
    out = np.where(cci <= -zone, -1, out)
    out = np.where(cci >= zone, 1, out)
    return out.astype(np.int8)


def rsi_bin_from_rsi(rsi: Array) -> Array:
    out = np.zeros(np.shape(rsi), dtype=np.int8)
    out = np.where(rsi > 50.0, 1, out)
    out = np.where(rsi < 50.0, -1, out)
    return out.astype(np.int8)


def rsi_dist50(rsi: Array) -> Array:
    return (rsi - 50.0) / 50.0


def pair_consensus(a: Array, b: Array) -> Array:
    """+1 if both +1, -1 if both -1, else 0."""
    out = np.zeros(np.shape(a), dtype=np.int8)
    out = np.where((a == 1) & (b == 1), 1, out)
    out = np.where((a == -1) & (b == -1), -1, out)
    return out.astype(np.int8)


def four_consensus(a: Array, b: Array, c: Array, d: Array) -> Array:
    out = np.zeros(np.shape(a), dtype=np.int8)
    out = np.where((a == 1) & (b == 1) & (c == 1) & (d == 1), 1, out)
    out = np.where((a == -1) & (b == -1) & (c == -1) & (d == -1), -1, out)
    return out.astype(np.int8)


def rsi_pullback_state(cons: Array, rsi3_bin_5m: Array, price_bin: Array) -> Array:
    out = np.zeros(np.shape(cons), dtype=np.int8)
    out = np.where((cons == 1) & (rsi3_bin_5m == -1) & (price_bin == -1), 1, out)
    out = np.where((cons == -1) & (rsi3_bin_5m == 1) & (price_bin == 1), -1, out)
    return out.astype(np.int8)


def rsi_pullback_state_rsi_only(cons: Array, rsi3_bin_5m: Array) -> Array:
    out = np.zeros(np.shape(cons), dtype=np.int8)
    out = np.where((cons == 1) & (rsi3_bin_5m == -1), 1, out)
    out = np.where((cons == -1) & (rsi3_bin_5m == 1), -1, out)
    return out.astype(np.int8)


def bbs_state_htf(close: Array, sma20: Array, mid: Array) -> Array:
    out = np.zeros(np.shape(close), dtype=np.int8)
    out = np.where((close > sma20) & (close > mid), 1, out)
    out = np.where((close < sma20) & (close < mid), -1, out)
    return out.astype(np.int8)


def bbs_state_5m(close: Array, sma20: Array, mid: Array) -> Array:
    out = np.zeros(np.shape(close), dtype=np.int8)
    out = np.where((close < mid) & (close > sma20), 1, out)
    out = np.where((close > mid) & (close < sma20), -1, out)
    return out.astype(np.int8)


def three_label(a: Array, b: Array, c: Array) -> Array:
    out = np.zeros(np.shape(a), dtype=np.int8)
    out = np.where((a == 1) & (b == 1) & (c == 1), 1, out)
    out = np.where((a == -1) & (b == -1) & (c == -1), -1, out)
    return out.astype(np.int8)


def ladder_point(line: Array, partner: Array, sma50: Array) -> Array:
    out = np.zeros(np.shape(line), dtype=np.int8)
    out = np.where((line > partner) & (line > sma50), 1, out)
    out = np.where((line < partner) & (line < sma50), -1, out)
    return out.astype(np.int8)


def ladder_bonus(
    p1_5: Array,
    p2_5: Array,
    p3_5: Array,
    p1_30: Array,
    p2_30: Array,
    p3_30: Array,
    p1_1h: Array,
    p2_1h: Array,
    p3_1h: Array,
) -> Array:
    htf_all_pos = ((p1_30 == 1) & (p2_30 == 1) & (p3_30 == 1)) | (
        (p1_1h == 1) & (p2_1h == 1) & (p3_1h == 1)
    )
    htf_all_neg = ((p1_30 == -1) & (p2_30 == -1) & (p3_30 == -1)) | (
        (p1_1h == -1) & (p2_1h == -1) & (p3_1h == -1)
    )
    pos = (p2_5 == 1) & (p3_5 == 1) & (p1_5 != 1) & htf_all_pos
    neg = (p2_5 == -1) & (p3_5 == -1) & (p1_5 != -1) & htf_all_neg
    out = np.zeros(np.shape(p1_5), dtype=np.int8)
    out = np.where(pos, 2, out)
    out = np.where(neg, -2, out)
    return out.astype(np.int8)


def safe_div(num: Array, den: Array) -> Array:
    """Divide; NaN when denominator is 0 or either side non-finite."""
    out = np.full(np.shape(num), np.nan, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[ok] = num[ok] / den[ok]
    return out
