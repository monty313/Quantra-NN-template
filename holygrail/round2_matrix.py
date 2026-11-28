"""Round 2 combined observation matrix. Observation only. No training here.

All Stage 1 blocks on 1m/5m/15m/30m/1h, 30m computed once, dual-support
consensus (s1 = 15m+30m with LTF 1m; s2 = 30m+1h with LTF 5m), TF prefixes.

Grain: last 1m of each 5m bar (same thin as Stage 1 train rows). Indicators
still use live_1m + developing HTF on the full 1m series, then we slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from holygrail.config import (
    CCI_PERIODS,
    ROUND2_SETS,
    ROUND2_TFS,
    RSI_PERIODS,
    STD_PERIODS,
    TF_FREQ,
)
from holygrail.features import _tf_indicators, prev5m_dir_at_decision
from holygrail.labels import structure_direction
from holygrail.round2_columns import ROUND2_FEATURE_COLUMNS
from holygrail.round2_gaps import GapReport, build_gap_report
from holygrail.rules import (
    four_consensus,
    ladder_bonus,
    pair_consensus,
    rbb_label,
    rsi_pullback_state,
    rsi_pullback_state_rsi_only,
    rsisma_label,
    s4hl_label,
    three_label,
)
from holygrail.timing import resample_ohlc

# Keys copied out of `_tf_indicators` so we can free the rest (raw SMA, CCI, …).
_TF_KEEP: tuple[str, ...] = tuple(
    [f"cci{p}_gap" for p in CCI_PERIODS]
    + [f"cci{p}_bin" for p in CCI_PERIODS]
    + [f"cci{p}_lvl" for p in CCI_PERIODS]
    + [f"std{p}_gap" for p in STD_PERIODS]
    + [f"std{p}_bin" for p in STD_PERIODS]
    + [f"sma_{pair}_{k}" for pair in ("ab", "bc", "cd") for k in ("bin", "gap")]
    + [f"rsi{p}_bin" for p in RSI_PERIODS]
    + [f"rsi{p}_dist50" for p in RSI_PERIODS]
    + ["price_sma20_bin", "price_sma20_gap"]
    + ["bbs_sma20_gap", "bbs_mid_gap", "bbs_pctb", "bbs_state"]
    + ["lad_p1", "lad_p2", "lad_p3", "lad_g1", "lad_g2", "lad_g3", "lad_50gap", "lad_score"]
    + ["s714_bin", "s714_gap"]
    + [f"rsisma{p}_{k}" for p in RSI_PERIODS for k in ("bin", "gap")]
    + ["rbb_pctb", "rbb_mid_gap", "rbb_mid_bin", "rbb_band_state"]
    + ["s4_hi_gap", "s4_lo_gap", "s4_state", "s20_gap", "s4rbb_pctb", "s4rbb_band_state"]
)


@dataclass
class Round2Table:
    """X is ROUND2_FEATURE_COLUMNS. y and close-to-close sit beside it, never in it."""

    timestamps: pd.DatetimeIndex  # 5m bar open = last-1m open floored to 5m
    last_1m: pd.DatetimeIndex
    decision_time: pd.DatetimeIndex
    X: pd.DataFrame
    valid_x: np.ndarray
    y_5m_dir: np.ndarray
    y_5m_dir_valid: np.ndarray
    y_5m_close: np.ndarray  # reference only — not the train label
    y_5m_close_valid: np.ndarray
    drop_lookback_gap: np.ndarray
    drop_short: np.ndarray
    drop_long_tail: np.ndarray
    row_incomplete: dict[str, np.ndarray]
    gap_report: GapReport


def _slim_tf(raw: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k in _TF_KEEP:
        out[k] = np.asarray(raw[k], dtype=np.float32)[mask]
    return out


def structure_direction_close(m1: pd.DataFrame, tf_key: str = "5m") -> tuple[np.ndarray, np.ndarray]:
    """sign(close(B_{t+1}) − close(B_t)). Ties and missing → invalid. Not y_5m_dir."""
    freq = TF_FREQ[tf_key]
    from holygrail.config import STRUCTURE_BAR_MINUTES

    bar_minutes = STRUCTURE_BAR_MINUTES[tf_key]
    completed = resample_ohlc(m1, freq)
    close_v = completed["close"].to_numpy(dtype=np.float64)
    idx = completed.index
    current_open = m1.index.floor(freq)
    next_open = current_open + pd.Timedelta(minutes=bar_minutes)
    loc_now = idx.get_indexer(current_open)
    loc_next = idx.get_indexer(next_open)
    have = (loc_now >= 0) & (loc_next >= 0)
    y = np.full(len(m1), np.nan, dtype=np.float64)
    valid = np.zeros(len(m1), dtype=bool)
    if have.any():
        move = close_v[loc_next[have]] - close_v[loc_now[have]]
        nonzero = move != 0.0
        y_have = np.full(have.sum(), np.nan, dtype=np.float64)
        y_have[nonzero] = np.sign(move[nonzero])
        y[have] = y_have
        valid[have] = nonzero
    return y, valid


def _put_per_tf(cols: dict[str, np.ndarray], tf: str, d: dict[str, np.ndarray]) -> None:
    for p in CCI_PERIODS:
        cols[f"{tf}_cci{p}_gap"] = d[f"cci{p}_gap"]
    for p in CCI_PERIODS:
        cols[f"{tf}_cci{p}_bin"] = d[f"cci{p}_bin"]
    for p in CCI_PERIODS:
        cols[f"{tf}_cci{p}_lvl"] = d[f"cci{p}_lvl"]
    for p in STD_PERIODS:
        cols[f"{tf}_std{p}_gap"] = d[f"std{p}_gap"]
    for p in STD_PERIODS:
        cols[f"{tf}_std{p}_bin"] = d[f"std{p}_bin"]
    for pair in ("ab", "bc", "cd"):
        cols[f"{tf}_sma_{pair}_bin"] = d[f"sma_{pair}_bin"]
        cols[f"{tf}_sma_{pair}_gap"] = d[f"sma_{pair}_gap"]
    for p in RSI_PERIODS:
        cols[f"{tf}_rsi{p}_bin"] = d[f"rsi{p}_bin"]
        cols[f"{tf}_rsi{p}_dist50"] = d[f"rsi{p}_dist50"]
    cols[f"{tf}_price_sma20_bin"] = d["price_sma20_bin"]
    cols[f"{tf}_price_sma20_gap"] = d["price_sma20_gap"]
    cols[f"{tf}_bbs_sma20_gap"] = d["bbs_sma20_gap"]
    cols[f"{tf}_bbs_mid_gap"] = d["bbs_mid_gap"]
    cols[f"{tf}_bbs_pctb"] = d["bbs_pctb"]
    cols[f"{tf}_bbs_state"] = d["bbs_state"]
    cols[f"{tf}_lad_p1"] = d["lad_p1"]
    cols[f"{tf}_lad_p2"] = d["lad_p2"]
    cols[f"{tf}_lad_p3"] = d["lad_p3"]
    cols[f"{tf}_lad_g1"] = d["lad_g1"]
    cols[f"{tf}_lad_g2"] = d["lad_g2"]
    cols[f"{tf}_lad_g3"] = d["lad_g3"]
    cols[f"{tf}_lad_50gap"] = d["lad_50gap"]
    cols[f"{tf}_lad_score"] = d["lad_score"]
    cols[f"{tf}_s714_bin"] = d["s714_bin"]
    cols[f"{tf}_s714_gap"] = d["s714_gap"]
    for p in RSI_PERIODS:
        cols[f"{tf}_rsisma{p}_bin"] = d[f"rsisma{p}_bin"]
        cols[f"{tf}_rsisma{p}_gap"] = d[f"rsisma{p}_gap"]
    cols[f"{tf}_rbb_pctb"] = d["rbb_pctb"]
    cols[f"{tf}_rbb_mid_gap"] = d["rbb_mid_gap"]
    cols[f"{tf}_rbb_mid_bin"] = d["rbb_mid_bin"]
    cols[f"{tf}_rbb_band_state"] = d["rbb_band_state"]
    cols[f"{tf}_s4_hi_gap"] = d["s4_hi_gap"]
    cols[f"{tf}_s4_lo_gap"] = d["s4_lo_gap"]
    cols[f"{tf}_s4_state"] = d["s4_state"]
    cols[f"{tf}_s20_gap"] = d["s20_gap"]
    cols[f"{tf}_s4rbb_pctb"] = d["s4rbb_pctb"]
    cols[f"{tf}_s4rbb_band_state"] = d["s4rbb_band_state"]


def _put_set_consensus(
    cols: dict[str, np.ndarray],
    set_id: str,
    tfs: dict[str, dict[str, np.ndarray]],
) -> None:
    spec = ROUND2_SETS[set_id]
    ltf, h1, h2 = spec["ltf"], spec["htf1"], spec["htf2"]
    cols[f"{set_id}_sma_ab_cons"] = pair_consensus(tfs[h1]["sma_ab_bin"], tfs[h2]["sma_ab_bin"])
    cols[f"{set_id}_sma_bc_cons"] = pair_consensus(tfs[h1]["sma_bc_bin"], tfs[h2]["sma_bc_bin"])
    cols[f"{set_id}_sma_cd_cons"] = pair_consensus(tfs[h1]["sma_cd_bin"], tfs[h2]["sma_cd_bin"])
    rsi_cons = four_consensus(
        tfs[h1]["rsi7_bin"],
        tfs[h1]["rsi14_bin"],
        tfs[h2]["rsi7_bin"],
        tfs[h2]["rsi14_bin"],
    )
    cols[f"{set_id}_rsi_htf_cons"] = rsi_cons
    cols[f"{set_id}_rsi_pullback_state"] = rsi_pullback_state(
        rsi_cons, tfs[ltf]["rsi3_bin"], tfs[ltf]["price_sma20_bin"]
    )
    cols[f"{set_id}_rsi_pullback_state_rsi_only"] = rsi_pullback_state_rsi_only(
        rsi_cons, tfs[ltf]["rsi3_bin"]
    )
    cols[f"{set_id}_bbs_label"] = three_label(
        tfs[ltf]["bbs_state"], tfs[h1]["bbs_state"], tfs[h2]["bbs_state"]
    )
    cols[f"{set_id}_lad_bonus"] = ladder_bonus(
        tfs[ltf]["lad_p1"],
        tfs[ltf]["lad_p2"],
        tfs[ltf]["lad_p3"],
        tfs[h1]["lad_p1"],
        tfs[h1]["lad_p2"],
        tfs[h1]["lad_p3"],
        tfs[h2]["lad_p1"],
        tfs[h2]["lad_p2"],
        tfs[h2]["lad_p3"],
    )
    cols[f"{set_id}_s714_label"] = three_label(
        tfs[ltf]["s714_bin"], tfs[h1]["s714_bin"], tfs[h2]["s714_bin"]
    )
    cols[f"{set_id}_rsisma_label"] = rsisma_label(
        tfs[ltf]["rsisma3_bin"],
        tfs[ltf]["rsisma7_bin"],
        tfs[ltf]["rsisma14_bin"],
        tfs[h1]["rsisma3_bin"],
        tfs[h1]["rsisma7_bin"],
        tfs[h1]["rsisma14_bin"],
        tfs[h2]["rsisma3_bin"],
        tfs[h2]["rsisma7_bin"],
        tfs[h2]["rsisma14_bin"],
    )
    cols[f"{set_id}_rbb_label"] = rbb_label(
        tfs[h1]["rbb_mid_bin"], tfs[h2]["rbb_mid_bin"], tfs[ltf]["rbb_band_state"]
    )
    cols[f"{set_id}_s4hl_label"] = s4hl_label(
        tfs[h1]["s4_state"],
        tfs[h2]["s4_state"],
        tfs[ltf]["s20_gap"],
        tfs[ltf]["s4rbb_band_state"],
    )


def persistence_acc(y: np.ndarray, valid: np.ndarray, last_1m: pd.DatetimeIndex) -> dict[str, float | int | None]:
    """Lag-1 persistence on consecutive 5m rows. Not a model score."""
    prev = np.roll(y, 1)
    prev_t = last_1m.to_numpy()
    dt = np.empty(len(last_1m), dtype="timedelta64[ns]")
    dt[0] = np.timedelta64("NaT", "ns")
    dt[1:] = prev_t[1:] - prev_t[:-1]
    consecutive = dt == np.timedelta64(5, "m")
    ok = valid & np.roll(valid, 1) & consecutive
    ok[0] = False
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "acc": None}
    acc = float(np.mean(y[ok] == prev[ok]))
    return {"n": n, "acc": acc}


def build_round2_matrix(m1: pd.DataFrame) -> Round2Table:
    """Build the combined matrix on last-1m-of-5m rows. Does not train."""
    gap = build_gap_report(m1)
    mask = gap.is_last_5m
    tfs: dict[str, dict[str, np.ndarray]] = {}
    for tf in ROUND2_TFS:
        raw = _tf_indicators(m1, tf)
        tfs[tf] = _slim_tf(raw, mask)
        del raw

    cols: dict[str, np.ndarray] = {}
    for tf in ROUND2_TFS:
        _put_per_tf(cols, tf, tfs[tf])
    for set_id in ROUND2_SETS:
        _put_set_consensus(cols, set_id, tfs)
    cols["prev5m_dir"] = np.asarray(prev5m_dir_at_decision(m1), dtype=np.float32)[mask]

    X = pd.DataFrame({name: cols[name] for name in ROUND2_FEATURE_COLUMNS})
    if list(X.columns) != ROUND2_FEATURE_COLUMNS:
        raise RuntimeError("round2 column order mismatch")
    X = X.astype(np.float32)
    valid_x = np.isfinite(X.to_numpy(dtype=np.float64)).all(axis=1)

    y_dir_full, v_dir_full = structure_direction(m1, "5m")
    y_cc_full, v_cc_full = structure_direction_close(m1, "5m")

    last_1m = m1.index[mask]
    if "decision_time" in m1.columns:
        decision = pd.DatetimeIndex(m1["decision_time"].to_numpy()[mask])
    else:
        decision = last_1m + pd.Timedelta(minutes=1)
    timestamps = last_1m.floor(TF_FREQ["5m"])

    return Round2Table(
        timestamps=timestamps,
        last_1m=last_1m,
        decision_time=decision,
        X=X,
        valid_x=valid_x,
        y_5m_dir=y_dir_full[mask],
        y_5m_dir_valid=v_dir_full[mask],
        y_5m_close=y_cc_full[mask],
        y_5m_close_valid=v_cc_full[mask],
        drop_lookback_gap=gap.drop_lookback_gap,
        drop_short=gap.drop_short,
        drop_long_tail=gap.drop_long_tail,
        row_incomplete=gap.row_incomplete,
        gap_report=gap,
    )


def _zero_move_count(
    m1: pd.DataFrame,
    last_1m: pd.DatetimeIndex,
    keep: np.ndarray,
    price: str,
) -> int:
    """Ties: both this and next 5m exist and the signed move is exact 0."""
    completed = resample_ohlc(m1, TF_FREQ["5m"])
    if price == "tp":
        series = (completed["high"] + completed["low"] + completed["close"]) / 3.0
    elif price == "close":
        series = completed["close"]
    else:
        raise ValueError(price)
    vals = series.to_numpy(dtype=np.float64)
    idx = completed.index
    current = last_1m.floor(TF_FREQ["5m"])
    nxt = current + pd.Timedelta(minutes=5)
    loc_now = idx.get_indexer(current)
    loc_next = idx.get_indexer(nxt)
    have = (loc_now >= 0) & (loc_next >= 0) & keep
    if not have.any():
        return 0
    move = vals[loc_next[have]] - vals[loc_now[have]]
    return int((move == 0.0).sum())


def count_report(table: Round2Table, m1: pd.DataFrame) -> dict:
    """Headline counts. No model fit."""
    drop = table.drop_lookback_gap
    keep = ~drop
    n_keep = int(keep.sum())
    n_drop = int(drop.sum())
    n_x_ok = int((keep & table.valid_x).sum())
    n_y_dir_ok = int((keep & table.y_5m_dir_valid).sum())
    n_y_cc_ok = int((keep & table.y_5m_close_valid).sum())
    n_dir_ties = _zero_move_count(m1, table.last_1m, keep, "tp")
    n_cc_ties = _zero_move_count(m1, table.last_1m, keep, "close")
    persist_dir = persistence_acc(table.y_5m_dir, table.y_5m_dir_valid & keep, table.last_1m)
    persist_cc = persistence_acc(table.y_5m_close, table.y_5m_close_valid & keep, table.last_1m)
    g = table.gap_report
    return {
        "feature_version": "2.0-step3",
        "n_columns": int(table.X.shape[1]),
        "n_5m_rows_before_gap_drop": int(len(table.X)),
        "n_dropped_lookback_gap": n_drop,
        "n_dropped_short_lookback": int(table.drop_short.sum()),
        "n_dropped_long_hole_tail": int(table.drop_long_tail.sum()),
        "n_dropped_short_only": int((table.drop_short & ~table.drop_long_tail).sum()),
        "n_dropped_long_only": int((table.drop_long_tail & ~table.drop_short).sum()),
        "n_dropped_both_rules": int((table.drop_short & table.drop_long_tail).sum()),
        "n_rows_after_gap_drop": n_keep,
        "n_rows_finite_X_after_gap_drop": n_x_ok,
        "n_y_5m_dir_valid_after_gap_drop": n_y_dir_ok,
        "n_y_5m_dir_ties_dropped": n_dir_ties,
        "n_y_5m_close_valid_after_gap_drop": n_y_cc_ok,
        "n_y_5m_close_ties_dropped": n_cc_ties,
        "y_5m_dir_persistence": persist_dir,
        "y_5m_close_persistence_reference": persist_cc,
        "y_5m_close_model_acc": None,  # no train this step
        "micro_gap_n": g.micro_gap_n,
        "session_break_n": g.session_break_n,
        "incomplete_completed_bars": g.incomplete_completed,
        "n_completed_bars": g.n_completed,
    }
