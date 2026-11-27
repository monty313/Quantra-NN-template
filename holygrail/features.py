"""Build the 128-column observation matrix X (spec §11). Raw prices never enter X.

Why 128 fixed order: later models (LightGBM / MLP / RL) must be able to swap
without renaming columns. Append future features after 128; never reorder.

Why no fill: a NaN means the row is not ready (warm-up, zero std, zero ATR).
Imputing 0 or the train median (adaptive-risk SimpleImputer) invents a fake
neutral state. Validity is a side flag, not a column in X.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from holygrail.asof import asof_atr, asof_cci, asof_rsi, asof_sma, asof_std, lookback
from holygrail.columns import FEATURE_COLUMNS
from holygrail.config import (
    ATR_PERIOD,
    BB_DEVIATION,
    BB_DDOF,
    BB_PERIOD,
    CCI_LEVEL_ZONE,
    CCI_PERIODS,
    RSI_PERIODS,
    SHIFT_BARS,
    STD_PERIODS,
    TF_FREQ,
)
from holygrail.indicators import cci, rolling_std, sma
from holygrail.rules import (
    bbs_state_5m,
    bbs_state_htf,
    cci_level,
    four_consensus,
    ladder_bonus,
    ladder_point,
    pair_consensus,
    rsi_bin_from_rsi,
    rsi_dist50,
    rsi_pullback_state,
    rsi_pullback_state_rsi_only,
    safe_div,
    sign_np,
    three_label,
)
from holygrail.timing import completed_and_dev


@dataclass
class FeatureTable:
    """X is 128 columns. Validity and ATR live beside it, never inside it."""

    timestamps: pd.DatetimeIndex  # 1m open times
    decision_time: pd.DatetimeIndex
    X: pd.DataFrame
    valid: np.ndarray
    atr_5m: np.ndarray  # for labels only


def _tf_indicators(m1: pd.DataFrame, tf_key: str) -> dict[str, np.ndarray]:
    completed, dev, bar_id_s = completed_and_dev(m1, tf_key)
    idx = bar_id_s.to_numpy()
    c = completed
    tp_c = c["tp"].to_numpy()
    close_c = c["close"].to_numpy()
    high_c = c["high"].to_numpy()
    low_c = c["low"].to_numpy()
    tp_d = dev["tp"].to_numpy()
    close_d = dev["close"].to_numpy()
    high_d = dev["high"].to_numpy()
    low_d = dev["low"].to_numpy()

    out: dict[str, np.ndarray] = {
        "close": close_d,
        "tp": tp_d,
        "atr": asof_atr(high_c, low_c, close_c, idx, high_d, low_d, ATR_PERIOD),
    }

    # CCI + SMA of CCI (gap) + bin vs c-2 + level
    for p in CCI_PERIODS:
        cci_c = cci(tp_c, p)
        cci_d = asof_cci(tp_c, idx, tp_d, p)
        sma_cci = asof_sma(cci_c, idx, cci_d, p)
        out[f"cci{p}"] = cci_d
        out[f"cci{p}_gap"] = cci_d - sma_cci
        out[f"cci{p}_bin"] = sign_np(cci_d - lookback(cci_c, idx, SHIFT_BARS))
        out[f"cci{p}_lvl"] = cci_level(cci_d, CCI_LEVEL_ZONE)

    for p in STD_PERIODS:
        std_d = asof_std(tp_c, idx, tp_d, p)
        std_c = rolling_std(tp_c, p)
        out[f"std{p}"] = std_d
        out[f"std{p}_gap"] = safe_div(std_d - lookback(std_c, idx, SHIFT_BARS), std_d)
        out[f"std{p}_bin"] = sign_np(std_d - lookback(std_c, idx, SHIFT_BARS))

    # SMA chain on Close
    a_c = sma(close_c, 5)
    b_c = sma(a_c, 5)
    c_sma_c = sma(b_c, 20)
    d_c = sma(close_c, 50)
    a_d = asof_sma(close_c, idx, close_d, 5)
    b_d = asof_sma(a_c, idx, a_d, 5)
    c_d = asof_sma(b_c, idx, b_d, 20)
    d_d = asof_sma(close_c, idx, close_d, 50)
    atr = out["atr"]
    out["sma_a"] = a_d
    out["sma_b"] = b_d
    out["sma_c"] = c_d
    out["sma_d"] = d_d
    out["sma_ab_bin"] = sign_np(a_d - b_d)
    out["sma_bc_bin"] = sign_np(b_d - c_d)
    out["sma_cd_bin"] = sign_np(c_d - d_d)
    out["sma_ab_gap"] = safe_div(a_d - b_d, atr)
    out["sma_bc_gap"] = safe_div(b_d - c_d, atr)
    out["sma_cd_gap"] = safe_div(c_d - d_d, atr)

    # RSI on typical price
    for p in RSI_PERIODS:
        rsi_d = asof_rsi(tp_c, idx, tp_d, p)
        out[f"rsi{p}"] = rsi_d
        out[f"rsi{p}_bin"] = rsi_bin_from_rsi(rsi_d)
        out[f"rsi{p}_dist50"] = rsi_dist50(rsi_d)

    # Price vs SMA20 of typical (5m block uses this; computed on every TF)
    sma20_tp_c = sma(tp_c, 20)
    sma20_tp_d = asof_sma(tp_c, idx, tp_d, 20)
    out["sma20_tp"] = sma20_tp_d
    out["price_sma20_bin"] = sign_np(tp_d - sma20_tp_d)
    out["price_sma20_gap"] = safe_div(tp_d - sma20_tp_d, atr)

    # Bollinger on Close, SMA20 of Close
    sma20_c = sma(close_c, 20)
    sma20_d = asof_sma(close_c, idx, close_d, 20)
    mid_d = asof_sma(close_c, idx, close_d, BB_PERIOD)
    std_d = asof_std(close_c, idx, close_d, BB_PERIOD, ddof=BB_DDOF)
    upper = mid_d + BB_DEVIATION * std_d
    lower = mid_d - BB_DEVIATION * std_d
    out["sma20_close"] = sma20_d
    out["bb_mid"] = mid_d
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bbs_sma20_gap"] = safe_div(close_d - sma20_d, atr)
    out["bbs_mid_gap"] = safe_div(close_d - mid_d, atr)
    width = upper - lower
    pctb = safe_div(close_d - lower, width)
    out["bbs_pctb"] = pctb
    if tf_key == "5m":
        out["bbs_state"] = bbs_state_5m(close_d, sma20_d, mid_d)
    else:
        out["bbs_state"] = bbs_state_htf(close_d, sma20_d, mid_d)

    # Ladder: SMA1 shift0 = close; SMA2/3/4 with shifts 1/2/3 on completed SMAs
    sma1 = close_d  # period 1, shift 0
    sma2_c = sma(close_c, 2)
    sma3_c = sma(close_c, 3)
    sma4_c = sma(close_c, 4)
    sma50_d = d_d  # period 50, no shift, developing
    sma2_s1 = lookback(sma2_c, idx, 1)
    sma3_s2 = lookback(sma3_c, idx, 2)
    sma4_s3 = lookback(sma4_c, idx, 3)
    out["lad_sma1"] = sma1
    out["lad_sma2"] = sma2_s1
    out["lad_sma3"] = sma3_s2
    out["lad_sma4"] = sma4_s3
    out["lad_sma50"] = sma50_d
    out["lad_p1"] = ladder_point(sma1, sma2_s1, sma50_d)
    out["lad_p2"] = ladder_point(sma2_s1, sma3_s2, sma50_d)
    out["lad_p3"] = ladder_point(sma3_s2, sma4_s3, sma50_d)
    out["lad_g1"] = safe_div(sma1 - sma2_s1, atr)
    out["lad_g2"] = safe_div(sma2_s1 - sma3_s2, atr)
    out["lad_g3"] = safe_div(sma3_s2 - sma4_s3, atr)
    out["lad_50gap"] = safe_div(sma1 - sma50_d, atr)
    out["lad_score"] = (
        out["lad_p1"].astype(np.int16) + out["lad_p2"].astype(np.int16) + out["lad_p3"].astype(np.int16)
    ).astype(np.int8)

    # sma714 on Close, no shifts
    sma7_d = asof_sma(close_c, idx, close_d, 7)
    sma14_d = asof_sma(close_c, idx, close_d, 14)
    out["sma7"] = sma7_d
    out["sma14"] = sma14_d
    out["s714_bin"] = sign_np(sma7_d - sma14_d)
    out["s714_gap"] = safe_div(sma7_d - sma14_d, atr)
    return out


def build_features(m1: pd.DataFrame) -> FeatureTable:
    """One shared code path for history and live. `m1` indexed by 1m open time."""
    tfs = {}
    for key in ("5m", "30m", "1h"):
        tfs[key] = _tf_indicators(m1, key)

    n = len(m1)
    cols: dict[str, np.ndarray] = {}

    def _put(name: str, arr: np.ndarray) -> None:
        cols[name] = arr

    prefix = {"5m": "ltf", "30m": "htf1", "1h": "htf2"}
    for tf, pre in prefix.items():
        d = tfs[tf]
        for p in CCI_PERIODS:
            _put(f"{pre}_cci{p}_gap", d[f"cci{p}_gap"])
        for p in CCI_PERIODS:
            _put(f"{pre}_cci{p}_bin", d[f"cci{p}_bin"])
        for p in CCI_PERIODS:
            _put(f"{pre}_cci{p}_lvl", d[f"cci{p}_lvl"])
        for p in STD_PERIODS:
            _put(f"{pre}_std{p}_gap", d[f"std{p}_gap"])
        for p in STD_PERIODS:
            _put(f"{pre}_std{p}_bin", d[f"std{p}_bin"])

    for tf, suf in (("5m", "5m"), ("30m", "30m"), ("1h", "1h")):
        d = tfs[tf]
        for pair in ("ab", "bc", "cd"):
            _put(f"sma_{pair}_bin_{suf}", d[f"sma_{pair}_bin"])
            _put(f"sma_{pair}_gap_{suf}", d[f"sma_{pair}_gap"])
    _put("sma_ab_cons", pair_consensus(tfs["30m"]["sma_ab_bin"], tfs["1h"]["sma_ab_bin"]))
    _put("sma_bc_cons", pair_consensus(tfs["30m"]["sma_bc_bin"], tfs["1h"]["sma_bc_bin"]))
    _put("sma_cd_cons", pair_consensus(tfs["30m"]["sma_cd_bin"], tfs["1h"]["sma_cd_bin"]))

    for tf, suf in (("5m", "5m"), ("30m", "30m"), ("1h", "1h")):
        d = tfs[tf]
        for p in RSI_PERIODS:
            _put(f"rsi{p}_bin_{suf}", d[f"rsi{p}_bin"])
            _put(f"rsi{p}_dist50_{suf}", d[f"rsi{p}_dist50"])
    rsi_cons = four_consensus(
        tfs["30m"]["rsi7_bin"],
        tfs["30m"]["rsi14_bin"],
        tfs["1h"]["rsi7_bin"],
        tfs["1h"]["rsi14_bin"],
    )
    _put("rsi_htf_cons", rsi_cons)
    price_bin = tfs["5m"]["price_sma20_bin"]
    _put(
        "rsi_pullback_state",
        rsi_pullback_state(rsi_cons, tfs["5m"]["rsi3_bin"], price_bin),
    )
    _put("ltf_price_sma20_bin_5m", price_bin)
    _put("ltf_price_sma20_gap_5m", tfs["5m"]["price_sma20_gap"])
    _put(
        "rsi_pullback_state_rsi_only",
        rsi_pullback_state_rsi_only(rsi_cons, tfs["5m"]["rsi3_bin"]),
    )

    for tf, suf in (("5m", "5m"), ("30m", "30m"), ("1h", "1h")):
        d = tfs[tf]
        _put(f"bbs_sma20_gap_{suf}", d["bbs_sma20_gap"])
        _put(f"bbs_mid_gap_{suf}", d["bbs_mid_gap"])
        _put(f"bbs_pctb_{suf}", d["bbs_pctb"])
        _put(f"bbs_state_{suf}", d["bbs_state"])
    _put("bbs_label", three_label(tfs["5m"]["bbs_state"], tfs["30m"]["bbs_state"], tfs["1h"]["bbs_state"]))

    for tf, suf in (("5m", "5m"), ("30m", "30m"), ("1h", "1h")):
        d = tfs[tf]
        _put(f"lad_p1_{suf}", d["lad_p1"])
        _put(f"lad_p2_{suf}", d["lad_p2"])
        _put(f"lad_p3_{suf}", d["lad_p3"])
        _put(f"lad_g1_{suf}", d["lad_g1"])
        _put(f"lad_g2_{suf}", d["lad_g2"])
        _put(f"lad_g3_{suf}", d["lad_g3"])
        _put(f"lad_50gap_{suf}", d["lad_50gap"])
        _put(f"lad_score_{suf}", d["lad_score"])
    _put(
        "lad_bonus",
        ladder_bonus(
            tfs["5m"]["lad_p1"],
            tfs["5m"]["lad_p2"],
            tfs["5m"]["lad_p3"],
            tfs["30m"]["lad_p1"],
            tfs["30m"]["lad_p2"],
            tfs["30m"]["lad_p3"],
            tfs["1h"]["lad_p1"],
            tfs["1h"]["lad_p2"],
            tfs["1h"]["lad_p3"],
        ),
    )

    for tf, suf in (("5m", "5m"), ("30m", "30m"), ("1h", "1h")):
        d = tfs[tf]
        _put(f"s714_bin_{suf}", d["s714_bin"])
        _put(f"s714_gap_{suf}", d["s714_gap"])
    _put("s714_label", three_label(tfs["5m"]["s714_bin"], tfs["30m"]["s714_bin"], tfs["1h"]["s714_bin"]))

    X = pd.DataFrame({name: cols[name] for name in FEATURE_COLUMNS}, index=m1.index)
    if list(X.columns) != FEATURE_COLUMNS:
        raise RuntimeError("column order mismatch")
    finite = np.isfinite(X.to_numpy(dtype=np.float64)).all(axis=1)
    valid = finite
    X = X.astype(np.float32)
    decision = pd.DatetimeIndex(m1["decision_time"]) if "decision_time" in m1.columns else m1.index + pd.Timedelta(minutes=1)
    return FeatureTable(
        timestamps=m1.index,
        decision_time=decision,
        X=X,
        valid=valid,
        atr_5m=tfs["5m"]["atr"],
    )
