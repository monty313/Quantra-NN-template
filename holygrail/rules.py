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


def two_of_three(a: Array, b: Array, c: Array, s: int) -> Array:
    """True where at least two of three bins equal `s`."""
    return (
        (a == s).astype(np.int8) + (b == s).astype(np.int8) + (c == s).astype(np.int8)
    ) >= 2


def rsisma_label(
    b3_5: Array,
    b7_5: Array,
    b14_5: Array,
    b3_30: Array,
    b7_30: Array,
    b14_30: Array,
    b3_1h: Array,
    b7_1h: Array,
    b14_1h: Array,
) -> Array:
    """+1 if 5m has ≥2 of 3 RSI-SMA bins +1 AND at least one HTF also does. −1 mirror. Else 0."""
    up5 = two_of_three(b3_5, b7_5, b14_5, 1)
    dn5 = two_of_three(b3_5, b7_5, b14_5, -1)
    up_htf = two_of_three(b3_30, b7_30, b14_30, 1) | two_of_three(b3_1h, b7_1h, b14_1h, 1)
    dn_htf = two_of_three(b3_30, b7_30, b14_30, -1) | two_of_three(b3_1h, b7_1h, b14_1h, -1)
    out = np.zeros(np.shape(b3_5), dtype=np.int8)
    out = np.where(up5 & up_htf, 1, out)
    out = np.where(dn5 & dn_htf, -1, out)
    return out.astype(np.int8)


def band_state(x: Array, upper: Array, lower: Array) -> Array:
    """+1 above upper band, −1 below lower, 0 otherwise (including equal)."""
    out = np.zeros(np.shape(x), dtype=np.int8)
    out = np.where(x > upper, 1, out)
    out = np.where(x < lower, -1, out)
    return out.astype(np.int8)


def rbb_label(mid_bin_30: Array, mid_bin_1h: Array, band_state_5m: Array) -> Array:
    """+1 both HTF mid bins +1 and 5m below lower band. −1 mirror. Else 0."""
    out = np.zeros(np.shape(mid_bin_30), dtype=np.int8)
    out = np.where((mid_bin_30 == 1) & (mid_bin_1h == 1) & (band_state_5m == -1), 1, out)
    out = np.where((mid_bin_30 == -1) & (mid_bin_1h == -1) & (band_state_5m == 1), -1, out)
    return out.astype(np.int8)


def s4_state(close: Array, sma4h: Array, sma4l: Array) -> Array:
    """+1 Close above both SMA4H and SMA4L, −1 below both, 0 between or equal."""
    out = np.zeros(np.shape(close), dtype=np.int8)
    out = np.where((close > sma4h) & (close > sma4l), 1, out)
    out = np.where((close < sma4h) & (close < sma4l), -1, out)
    return out.astype(np.int8)


def s4hl_label(s4_30: Array, s4_1h: Array, s20_gap_5m: Array, band_state_5m: Array) -> Array:
    """+1 both HTF s4_state +1, 5m s20_gap > 0, 5m RSI below lower band. −1 mirror. Else 0."""
    out = np.zeros(np.shape(s4_30), dtype=np.int8)
    out = np.where((s4_30 == 1) & (s4_1h == 1) & (s20_gap_5m > 0) & (band_state_5m == -1), 1, out)
    out = np.where((s4_30 == -1) & (s4_1h == -1) & (s20_gap_5m < 0) & (band_state_5m == 1), -1, out)
    return out.astype(np.int8)


def safe_div(num: Array, den: Array) -> Array:
    """Divide; NaN when denominator is 0 or either side non-finite."""
    out = np.full(np.shape(num), np.nan, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[ok] = num[ok] / den[ok]
    return out
