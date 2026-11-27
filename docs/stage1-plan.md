# Stage 1 market-state model — plan

Status: code + tests green on `cursor/stage1-market-state-a365`. Full EURUSD train/eval finishing; why-notes live in this file, `stage1-why-notes.md`, and `holygrail/` module docs.

## Goal

Learn market structure from a 128-column observation matrix `X` plus two 3-class targets (`y_15m`, `y_1h`). Outputs are calibrated state probabilities, not trades.

## Layers

1. Feature pipeline (`holygrail/`): 1m MT5 → developing 5m/30m/1h → 128-col `X` + validity.
2. Labels: `y_15m` / `y_1h` stored separately.
3. Pluggable policy: `fit(X, y)` / `predict_proba(X)` / `explain(X)`.

## Data (Step 0)

Google Drive folder downloaded. Four MT5 M1 exports, tab-separated `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>`. Timestamps are bar **open** times, naive (no timezone in file). Not UTC: no Sunday session; EURUSD week is Mon ~00:00–Fri ~23:59 with ~48h weekend gaps. Primary train symbol: **EURUSD** (~2.0M bars, 2021-01-13 → 2026-05-27).

## Spec decisions (lead defaults)

Recorded in the results report. Highlights: `ddof=0`, Wilder ATR, Wilder RSI (SMA seed), SMA shift = bar-index offset, timezone = naive broker-server (do not convert), `live_1m` + developing HTF.

## Build order

- Step 1–4: resample, indicators, `X`, labels
- Step 5–6: 70/30 time split, purge/embargo, RobustScaler on 46 gap cols
- Step 7: T1–T44
- Then models (majority, logistic, LightGBM, shallow MLP), calibration, regime slices, SHAP / permutation / block ablation

## Out of scope

Entries, exits, sizing, PnL, Stage 2.

## Lessons from prior repos

Studied monty313's trading/ML repos (skipped med-reminder). Goal: do not copy patterns that made earlier bots look good and then fail.

**Keep (these were the good habits):**
- Chronological split + purge of the label horizon (`adaptive-risk-ml-trader` train.py). Stage 1 also adds a 60-minute embargo, which that repo mostly skipped.
- Causality tests: mutating a future HTF bar must not change earlier rows (`test_feature_causality.py`). Same idea as our T3/T4/T25.
- Never put raw prices in the model (`MARK_TRADING_MIND` invariance rule). We only use prices inside indicators/labels.
- Observe-only until a later stage (`quant-reasoning`). Stage 1 outputs probabilities, not orders.
- Do not invent missing indicator settings (`quant-reasoning` D02/D03). Timezone stays naive broker-server; MT5 methods are named and tested, not guessed silently.
- Drop incomplete rows rather than inventing values. `svm-ftmo` at least used `dropna`. We go further: flag invalid, never fill.

**Do not copy:**
- Quantra template: accuracy as the score, binary sigmoid, daily BAC with UTC stamps, Keras net locked in. That course is a starting folder, not the model.
- `class_weight="balanced"` and median impute (`adaptive-risk-ml-trader`). Spec forbids rebalancing and filling NaNs with neutrals. Invalid rows are dropped.
- Scaling every column (`svm-ftmo` StandardScaler on all 114 features). Only the 46 gap columns get RobustScaler. Bins/levels stay raw.
- `TimeSeriesSplit` with no embargo (`svm-ftmo`). Overlapping 15m/1h labels leak across the cut without purge+embargo.
- Filling CCI when std=0 with 0.0 (`adaptive-risk` cci). Zero std means the row is invalid.
- ATR as pandas `ewm` from bar 0 with no SMA seed. MT5 iATR seeds with a simple mean of the first 14 TRs, then Wilder. We do that.
- Forward-fill HTF onto LTF (`svm-ftmo`). That copies a finished higher-timeframe bar into earlier snapshots. Spec forbids it. We rebuild developing 5m/30m/1h from completed 1m only.
- Empty / auto-README shells (`nn-ftmo-trading-bot`, `LSTM-Gold-Momentum-Hunter`, many `HANDOFF-2026-11-07.md` files). A README that says `python main.py` when there is no `main.py` is a prior mistake. Ours has real scripts and tests.
- Mixing HolyGrail 128-col spec with `regimes` StdDev 20/30 or Camillion 114-col stacks. One spec, one matrix.
- Treating Kronos entropy as a feature after it failed the 15-bar move test (`MARK_TRADING_MIND`). We do not add extra indicators.
- Placeholder nets with random weights shipped as "the model". MLP is trained with early stopping on time-ordered val; if it loses to logistic on test, we say so.
- PnL / FTMO / live MT5 bridges in Stage 1 (`svm-ftmo` bot.py, `the-truth`). Out of scope.

**Why this matters for Mark reading the code:** every surprising choice (ddof=0, Wilder seed, no fill, 70/30 by row time, scaler only on gaps) is there to avoid a mistake that already showed up in one of these repos.
