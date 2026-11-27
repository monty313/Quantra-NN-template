"""Fixed 128-column order and the 46 RobustScaler gap columns. Spec v1.0 §11 and §13."""

from __future__ import annotations

FEATURE_VERSION = "1.0"

FEATURE_COLUMNS: list[str] = [
    # Block 1 CCI+Std, LTF 5m
    "ltf_cci10_gap",
    "ltf_cci30_gap",
    "ltf_cci100_gap",
    "ltf_cci10_bin",
    "ltf_cci30_bin",
    "ltf_cci100_bin",
    "ltf_cci10_lvl",
    "ltf_cci30_lvl",
    "ltf_cci100_lvl",
    "ltf_std7_gap",
    "ltf_std14_gap",
    "ltf_std7_bin",
    "ltf_std14_bin",
    # HTF1 30m
    "htf1_cci10_gap",
    "htf1_cci30_gap",
    "htf1_cci100_gap",
    "htf1_cci10_bin",
    "htf1_cci30_bin",
    "htf1_cci100_bin",
    "htf1_cci10_lvl",
    "htf1_cci30_lvl",
    "htf1_cci100_lvl",
    "htf1_std7_gap",
    "htf1_std14_gap",
    "htf1_std7_bin",
    "htf1_std14_bin",
    # HTF2 1h
    "htf2_cci10_gap",
    "htf2_cci30_gap",
    "htf2_cci100_gap",
    "htf2_cci10_bin",
    "htf2_cci30_bin",
    "htf2_cci100_bin",
    "htf2_cci10_lvl",
    "htf2_cci30_lvl",
    "htf2_cci100_lvl",
    "htf2_std7_gap",
    "htf2_std14_gap",
    "htf2_std7_bin",
    "htf2_std14_bin",
    # Block 2 SMA chain
    "sma_ab_bin_5m",
    "sma_ab_gap_5m",
    "sma_bc_bin_5m",
    "sma_bc_gap_5m",
    "sma_cd_bin_5m",
    "sma_cd_gap_5m",
    "sma_ab_bin_30m",
    "sma_ab_gap_30m",
    "sma_bc_bin_30m",
    "sma_bc_gap_30m",
    "sma_cd_bin_30m",
    "sma_cd_gap_30m",
    "sma_ab_bin_1h",
    "sma_ab_gap_1h",
    "sma_bc_bin_1h",
    "sma_bc_gap_1h",
    "sma_cd_bin_1h",
    "sma_cd_gap_1h",
    "sma_ab_cons",
    "sma_bc_cons",
    "sma_cd_cons",
    # Block 3 RSI
    "rsi3_bin_5m",
    "rsi3_dist50_5m",
    "rsi7_bin_5m",
    "rsi7_dist50_5m",
    "rsi14_bin_5m",
    "rsi14_dist50_5m",
    "rsi3_bin_30m",
    "rsi3_dist50_30m",
    "rsi7_bin_30m",
    "rsi7_dist50_30m",
    "rsi14_bin_30m",
    "rsi14_dist50_30m",
    "rsi3_bin_1h",
    "rsi3_dist50_1h",
    "rsi7_bin_1h",
    "rsi7_dist50_1h",
    "rsi14_bin_1h",
    "rsi14_dist50_1h",
    "rsi_htf_cons",
    "rsi_pullback_state",
    # Block 4
    "ltf_price_sma20_bin_5m",
    "ltf_price_sma20_gap_5m",
    "rsi_pullback_state_rsi_only",
    # Block 5 bbsma
    "bbs_sma20_gap_5m",
    "bbs_mid_gap_5m",
    "bbs_pctb_5m",
    "bbs_state_5m",
    "bbs_sma20_gap_30m",
    "bbs_mid_gap_30m",
    "bbs_pctb_30m",
    "bbs_state_30m",
    "bbs_sma20_gap_1h",
    "bbs_mid_gap_1h",
    "bbs_pctb_1h",
    "bbs_state_1h",
    "bbs_label",
    # Block 6 ladder
    "lad_p1_5m",
    "lad_p2_5m",
    "lad_p3_5m",
    "lad_g1_5m",
    "lad_g2_5m",
    "lad_g3_5m",
    "lad_50gap_5m",
    "lad_score_5m",
    "lad_p1_30m",
    "lad_p2_30m",
    "lad_p3_30m",
    "lad_g1_30m",
    "lad_g2_30m",
    "lad_g3_30m",
    "lad_50gap_30m",
    "lad_score_30m",
    "lad_p1_1h",
    "lad_p2_1h",
    "lad_p3_1h",
    "lad_g1_1h",
    "lad_g2_1h",
    "lad_g3_1h",
    "lad_50gap_1h",
    "lad_score_1h",
    "lad_bonus",
    # Block 7 sma714
    "s714_bin_5m",
    "s714_gap_5m",
    "s714_bin_30m",
    "s714_gap_30m",
    "s714_bin_1h",
    "s714_gap_1h",
    "s714_label",
]

assert len(FEATURE_COLUMNS) == 128, len(FEATURE_COLUMNS)
assert len(set(FEATURE_COLUMNS)) == 128

SCALER_COLUMNS: list[str] = [
    "ltf_cci10_gap",
    "ltf_cci30_gap",
    "ltf_cci100_gap",
    "ltf_std7_gap",
    "ltf_std14_gap",
    "htf1_cci10_gap",
    "htf1_cci30_gap",
    "htf1_cci100_gap",
    "htf1_std7_gap",
    "htf1_std14_gap",
    "htf2_cci10_gap",
    "htf2_cci30_gap",
    "htf2_cci100_gap",
    "htf2_std7_gap",
    "htf2_std14_gap",
    "sma_ab_gap_5m",
    "sma_bc_gap_5m",
    "sma_cd_gap_5m",
    "sma_ab_gap_30m",
    "sma_bc_gap_30m",
    "sma_cd_gap_30m",
    "sma_ab_gap_1h",
    "sma_bc_gap_1h",
    "sma_cd_gap_1h",
    "ltf_price_sma20_gap_5m",
    "bbs_sma20_gap_5m",
    "bbs_mid_gap_5m",
    "bbs_sma20_gap_30m",
    "bbs_mid_gap_30m",
    "bbs_sma20_gap_1h",
    "bbs_mid_gap_1h",
    "lad_g1_5m",
    "lad_g2_5m",
    "lad_g3_5m",
    "lad_50gap_5m",
    "lad_g1_30m",
    "lad_g2_30m",
    "lad_g3_30m",
    "lad_50gap_30m",
    "lad_g1_1h",
    "lad_g2_1h",
    "lad_g3_1h",
    "lad_50gap_1h",
    "s714_gap_5m",
    "s714_gap_30m",
    "s714_gap_1h",
]

assert len(SCALER_COLUMNS) == 46, len(SCALER_COLUMNS)
assert all(c in FEATURE_COLUMNS for c in SCALER_COLUMNS)

# Explainability blocks (column names).
BLOCKS: dict[str, list[str]] = {
    "cci": [c for c in FEATURE_COLUMNS if "_cci" in c],
    "std": [c for c in FEATURE_COLUMNS if "_std" in c],
    "sma_chain": [c for c in FEATURE_COLUMNS if c.startswith("sma_")],
    "rsi_pullback": [
        c
        for c in FEATURE_COLUMNS
        if c.startswith("rsi") or c == "rsi_pullback_state"
    ]
    + ["rsi_pullback_state_rsi_only"],
    "price_vs_sma20": [
        "ltf_price_sma20_bin_5m",
        "ltf_price_sma20_gap_5m",
    ],
    "bbsma": [c for c in FEATURE_COLUMNS if c.startswith("bbs_")],
    "ladder": [c for c in FEATURE_COLUMNS if c.startswith("lad_")],
    "sma714": [c for c in FEATURE_COLUMNS if c.startswith("s714_")],
}

# rsi_pullback listed column 83 twice if we concatenate carelessly — keep unique order.
_seen: set[str] = set()
_rsi: list[str] = []
for name in BLOCKS["rsi_pullback"]:
    if name not in _seen:
        _seen.add(name)
        _rsi.append(name)
BLOCKS["rsi_pullback"] = _rsi

assert sum(len(v) for v in BLOCKS.values()) == 128, {k: len(v) for k, v in BLOCKS.items()}
