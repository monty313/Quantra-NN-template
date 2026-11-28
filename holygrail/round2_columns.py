"""Round 2 combined-matrix column order.

Stage 1 `FEATURE_COLUMNS` (180) is frozen. This list is a new matrix:
every existing block on 1m, 5m, 15m, 30m, 1h with TF prefixes; 30m once;
dual-support consensus for Set 1 (15m+30m, LTF 1m) and Set 2 (30m+1h, LTF 5m).

Why prefixes not ltf_/htf1_/htf2_: those names assumed one LTF. Combined
has two Speed TFs (1m and 5m). Prefix = the TF key (`1m_cci10_gap`).
"""

from __future__ import annotations

from holygrail.config import CCI_PERIODS, ROUND2_SETS, ROUND2_TFS, RSI_PERIODS, STD_PERIODS

ROUND2_FEATURE_VERSION = "2.0-step3"

_SETS = tuple(ROUND2_SETS.keys())


def _per_tf_block1(tf: str) -> list[str]:
    gaps = [f"{tf}_cci{p}_gap" for p in CCI_PERIODS]
    bins = [f"{tf}_cci{p}_bin" for p in CCI_PERIODS]
    lvls = [f"{tf}_cci{p}_lvl" for p in CCI_PERIODS]
    std_gaps = [f"{tf}_std{p}_gap" for p in STD_PERIODS]
    std_bins = [f"{tf}_std{p}_bin" for p in STD_PERIODS]
    return gaps + bins + lvls + std_gaps + std_bins


def _per_tf_block2(tf: str) -> list[str]:
    out: list[str] = []
    for pair in ("ab", "bc", "cd"):
        out.append(f"{tf}_sma_{pair}_bin")
        out.append(f"{tf}_sma_{pair}_gap")
    return out


def _per_tf_block3(tf: str) -> list[str]:
    out: list[str] = []
    for p in RSI_PERIODS:
        out.append(f"{tf}_rsi{p}_bin")
        out.append(f"{tf}_rsi{p}_dist50")
    return out


def _per_tf_block4(tf: str) -> list[str]:
    return [f"{tf}_price_sma20_bin", f"{tf}_price_sma20_gap"]


def _per_tf_block5(tf: str) -> list[str]:
    return [
        f"{tf}_bbs_sma20_gap",
        f"{tf}_bbs_mid_gap",
        f"{tf}_bbs_pctb",
        f"{tf}_bbs_state",
    ]


def _per_tf_block6(tf: str) -> list[str]:
    return [
        f"{tf}_lad_p1",
        f"{tf}_lad_p2",
        f"{tf}_lad_p3",
        f"{tf}_lad_g1",
        f"{tf}_lad_g2",
        f"{tf}_lad_g3",
        f"{tf}_lad_50gap",
        f"{tf}_lad_score",
    ]


def _per_tf_block7(tf: str) -> list[str]:
    return [f"{tf}_s714_bin", f"{tf}_s714_gap"]


def _per_tf_block8(tf: str) -> list[str]:
    out: list[str] = []
    for p in RSI_PERIODS:
        out.append(f"{tf}_rsisma{p}_bin")
        out.append(f"{tf}_rsisma{p}_gap")
    return out


def _per_tf_block9(tf: str) -> list[str]:
    return [
        f"{tf}_rbb_pctb",
        f"{tf}_rbb_mid_gap",
        f"{tf}_rbb_mid_bin",
        f"{tf}_rbb_band_state",
    ]


def _per_tf_block10(tf: str) -> list[str]:
    return [
        f"{tf}_s4_hi_gap",
        f"{tf}_s4_lo_gap",
        f"{tf}_s4_state",
        f"{tf}_s20_gap",
        f"{tf}_s4rbb_pctb",
        f"{tf}_s4rbb_band_state",
    ]


PER_TF_COLUMNS: dict[str, list[str]] = {}
for _tf in ROUND2_TFS:
    PER_TF_COLUMNS[_tf] = (
        _per_tf_block1(_tf)
        + _per_tf_block2(_tf)
        + _per_tf_block3(_tf)
        + _per_tf_block4(_tf)
        + _per_tf_block5(_tf)
        + _per_tf_block6(_tf)
        + _per_tf_block7(_tf)
        + _per_tf_block8(_tf)
        + _per_tf_block9(_tf)
        + _per_tf_block10(_tf)
    )

N_PER_TF = len(PER_TF_COLUMNS["5m"])
assert all(len(v) == N_PER_TF for v in PER_TF_COLUMNS.values()), {
    k: len(v) for k, v in PER_TF_COLUMNS.items()
}


def _build_round2_columns() -> list[str]:
    """Block-major, TF order 1m→5m→15m→30m→1h, then s1 consensus then s2."""
    cols: list[str] = []
    for tf in ROUND2_TFS:
        cols.extend(_per_tf_block1(tf))
    for tf in ROUND2_TFS:
        cols.extend(_per_tf_block2(tf))
    for s in _SETS:
        for pair in ("ab", "bc", "cd"):
            cols.append(f"{s}_sma_{pair}_cons")
    for tf in ROUND2_TFS:
        cols.extend(_per_tf_block3(tf))
    for s in _SETS:
        cols.append(f"{s}_rsi_htf_cons")
        cols.append(f"{s}_rsi_pullback_state")
    for tf in ROUND2_TFS:
        cols.extend(_per_tf_block4(tf))
    for s in _SETS:
        cols.append(f"{s}_rsi_pullback_state_rsi_only")
    for tf in ROUND2_TFS:
        cols.extend(_per_tf_block5(tf))
    for s in _SETS:
        cols.append(f"{s}_bbs_label")
    for tf in ROUND2_TFS:
        cols.extend(_per_tf_block6(tf))
    for s in _SETS:
        cols.append(f"{s}_lad_bonus")
    for tf in ROUND2_TFS:
        cols.extend(_per_tf_block7(tf))
    for s in _SETS:
        cols.append(f"{s}_s714_label")
    for tf in ROUND2_TFS:
        cols.extend(_per_tf_block8(tf))
    for s in _SETS:
        cols.append(f"{s}_rsisma_label")
    for tf in ROUND2_TFS:
        cols.extend(_per_tf_block9(tf))
    for s in _SETS:
        cols.append(f"{s}_rbb_label")
    for tf in ROUND2_TFS:
        cols.extend(_per_tf_block10(tf))
    for s in _SETS:
        cols.append(f"{s}_s4hl_label")
    cols.append("prev5m_dir")
    return cols


ROUND2_FEATURE_COLUMNS: list[str] = _build_round2_columns()

N_CONSENSUS_PER_SET = 12  # 3 sma cons + rsi_htf + pullback + rsi_only + bbs + lad + s714 + rsisma + rbb + s4hl
N_ROUND2_COLUMNS = N_PER_TF * len(ROUND2_TFS) + N_CONSENSUS_PER_SET * len(_SETS) + 1

assert len(ROUND2_FEATURE_COLUMNS) == N_ROUND2_COLUMNS, (
    len(ROUND2_FEATURE_COLUMNS),
    N_ROUND2_COLUMNS,
)
assert len(set(ROUND2_FEATURE_COLUMNS)) == len(ROUND2_FEATURE_COLUMNS)
assert ROUND2_FEATURE_COLUMNS[-1] == "prev5m_dir"
# 30m once: per-TF 30m block appears once, not s1_30m_ / s2_30m_.
assert sum(c.startswith("30m_") for c in ROUND2_FEATURE_COLUMNS) == N_PER_TF
assert sum(c.startswith("s1_") for c in ROUND2_FEATURE_COLUMNS) == N_CONSENSUS_PER_SET
assert sum(c.startswith("s2_") for c in ROUND2_FEATURE_COLUMNS) == N_CONSENSUS_PER_SET
# Dual-support pairs exist.
assert "s1_sma_ab_cons" in ROUND2_FEATURE_COLUMNS
assert "s2_sma_ab_cons" in ROUND2_FEATURE_COLUMNS
assert "s1_rsi_htf_cons" in ROUND2_FEATURE_COLUMNS
assert "s2_rsi_htf_cons" in ROUND2_FEATURE_COLUMNS


def _is_scaler_col(name: str) -> bool:
    """Unbounded gaps only. Binaries, %b, states, consensus, prev5m_dir stay raw."""
    if name == "prev5m_dir":
        return False
    if name.endswith("_gap") or name.endswith("50gap"):
        return True
    # lad_g1 / lad_g2 / lad_g3
    if "_lad_g" in name:
        return True
    return False


ROUND2_SCALER_COLUMNS: list[str] = [c for c in ROUND2_FEATURE_COLUMNS if _is_scaler_col(c)]
assert "prev5m_dir" not in ROUND2_SCALER_COLUMNS
assert all(c in ROUND2_FEATURE_COLUMNS for c in ROUND2_SCALER_COLUMNS)
assert not any("pctb" in c for c in ROUND2_SCALER_COLUMNS)
assert not any(c.endswith("_bin") for c in ROUND2_SCALER_COLUMNS)
# 23 gap cols per TF × 5 TFs (price_sma20_gap is on every TF in Round 2).
assert len(ROUND2_SCALER_COLUMNS) == 23 * len(ROUND2_TFS), len(ROUND2_SCALER_COLUMNS)
