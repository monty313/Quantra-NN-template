"""Frozen Stage 1 settings.

Lead defaults for spec MISSING/VERIFY (Mark asked us not to stall):
- timezone: naive broker-server open times. Files have no zone. Not UTC
  (no Sunday FX session). Do not convert.
- std/Bollinger ddof=0: MT5 iStdDev / iBands divide by n, not n-1.
- ATR/RSI: Wilder with SMA seed, matching iATR / iRSI, not pandas ewm-from-zero
  (that was the adaptive-risk-ml-trader shortcut).
- SMA shift: MT5 bar-index offset after the SMA (iMA `shift`).
Why freeze here: one place Mark can read instead of hunting magic numbers.
"""

from __future__ import annotations

from holygrail.columns import FEATURE_VERSION

# Timing (spec §2). User delegated live_1m + developing HTF.
SNAPSHOT_MODE = "live_1m"
HTF_BAR_MODE = "developing"
TIMEFRAME_SET = {"ltf": "5min", "htf1": "30min", "htf2": "1h"}
TF_FREQ = {"5m": "5min", "30m": "30min", "1h": "1h"}
TF_PREFIX = {"5m": "ltf", "30m": "htf1", "1h": "htf2"}

# Timezone: files carry no zone. Treat as naive broker-server open times.
# Do NOT convert to UTC. See Step 0 in the results report.
TIMEZONE = "broker_server_naive"

# Indicators
CCI_PERIODS = (10, 30, 100)
CCI_CONSTANT = 0.015
CCI_LEVEL_ZONE = 50.0
STD_PERIODS = (7, 14)
STD_DDOF = 0  # MT5 iStdDev / iBands population divisor (T2, T36)
SHIFT_BARS = 2
ATR_PERIOD = 14
ATR_METHOD = "wilder"  # MT5 iATR (T12)
RSI_PERIODS = (3, 7, 14)
RSI_METHOD = "wilder"  # MT5 iRSI: SMA seed then Wilder (T16)
BB_PERIOD = 5
BB_DEVIATION = 0.5
BB_DDOF = 0  # same population std as MT5 iBands (price BB and RSI BB)
LADDER_SMAS = ((1, 0), (2, 1), (3, 2), (4, 3))  # (period, shift)
SMA50_SHIFT = 0

# Phase B (rsi_sma_bb_feature_specs_v1.0). RSI/SMA/BB on Close, developing asof.
RSISMA_SMA_PERIOD = 5
RSI_BB_PERIOD = 4
RSI_BB_BB_PERIOD = 20
RSI_BB_DEVIATION = 1.0
SMA4_PERIOD = 4

# Labels
# Old 1m-tick heads (kept as unused comparison columns): 15m / 1h vs 0.5× 5m ATR.
LABEL_HORIZONS_MIN = (15, 60)
LABEL_ATR_FRACTION = 0.5
# Stage-1 training heads: sign of next-closed HTF typical-price change.
# No ATR size fence. Exact-zero move is invalid (dropped), not class 0.
STRUCTURE_HEADS = ("5m", "30m", "1h")
STRUCTURE_BAR_MINUTES = {"5m": 5, "30m": 30, "1h": 60}
N_CLASSES = 2
CLASS_LABELS = (-1, 1)
# 1h structure: current 1h remainder + next 1h ≤ 120m. Tightest of the three heads.
STRUCTURE_PURGE_MINUTES = 120

# Split / scaler
TRAIN_FRACTION = 0.70
EMBARGO_MINUTES = 60
VAL_FRACTION_OF_TRAIN = 0.20  # walk-forward tail of the train span
SCALER_QUANTILE_RANGE = (25.0, 75.0)

# Warm-up: slow CCI pair needs 199 TF bars; binaries need c-2.
WARMUP_BARS = 199
WARMUP_LOOKBACK = 2

FEATURE_VERSION = FEATURE_VERSION
