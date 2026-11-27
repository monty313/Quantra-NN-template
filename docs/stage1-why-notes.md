# Stage 1 — why the code looks like this

Mark: read this next to the repo. Short reasons, not a second spec.

## Three layers, on purpose

1. `holygrail/` builds X and validity from 1m candles. Same function for history and live.
2. `holygrail/labels.py` builds y_15m / y_1h in a **separate** table. Targets never enter X or the scaler.
3. `holygrail/policy.py` is `fit` / `predict_proba` / `explain`. LightGBM, logistic, MLP, or a later RL head can swap without touching features.

Quantra's Keras file is not the network. It is a course template with accuracy + sigmoid. We did not lock to it.

## Time and timezone

MT5 CSVs have `<DATE>` + `<TIME>`, no zone. They are bar **open** times.

- Decision time t = open + 1 minute (the close of that 1m candle).
- We do **not** convert to UTC. Forex UTC would show a Sunday evening session. These files have no Sunday bars, Monday ~00:00, Friday ~23:59. That is broker server time.

`quant-reasoning` already flagged "do not infer timezone from price patterns and then treat it as fact." We report the evidence and keep timestamps naive.

## Developing bars (why not resample-and-ffill)

`svm-ftmo` forward-filled 30M/4H onto 5M. That pastes a finished HTF close into every 1m snapshot inside the bar — look-ahead of the bar's own future.

Spec: at 10:06 the current 5m bar is the 10:05 bar with **one** 1m candle. Indicators use that incomplete bar as the latest point. Previous 5m bars are frozen.

## Invalid rows

Warm-up, NaN, zero ATR, zero std, missing future candle → flag and drop.

We do **not** `fillna(0)` or impute medians. `adaptive-risk-ml-trader` imputes median so the sklearn pipeline never sees NaN. That turns "I don't know" into a fake neutral and leaks the train median into live. Spec forbids it.

## Scaler

RobustScaler, 25–75, **training rows only**, **46 gap columns only**.

Why not StandardScaler on everything (`svm-ftmo`): bins are -1/0/+1. Scaling them destroys the meaning and mixes magnitude with sign. Binaries are computed from **raw** values (T8) so a scaled gap below the train median can still have binary +1.

IQR of 0 → scale 1, no clipping.

## Labels

move = TP(t+h) − TP(t). Threshold = 0.5 × 5m ATR14 at t.

No candle at exactly t+h → missing, not 0. Calling a weekend gap "flat" was the easy cheat; T27 guards it.

## Split

Oldest 70% of 1m history by row order. Purge train rows whose 1h window crosses the cutoff. First test row ≥ 60 minutes after cutoff. Val is the time-ordered tail of train with the same purge/embargo. Never shuffled.

`svm-ftmo` used TimeSeriesSplit (good) but no embargo (bad for overlapping 15m labels).

No class_weight. Spec: report counts, do not rebalance. `adaptive-risk` balanced classes — that makes logloss and calibration lie.

## Overlap

Consecutive 1m rows share 14/15 of the 15m window. We train on every 5th row and also compare sample weights 1/15. Effective N is smaller than row count. Do not read 1.1M rows as 1.1M independent samples.

## Models

- Majority: constant train class frequencies (fair logloss baseline).
- Logistic: L2 multinomial, no rebalance.
- LightGBM: shallow, early stop on time-ordered val.
- MLP: 128→64→32→3, dropout, weight decay, **not** Quantra's 5-layer sigmoid tower.

If MLP loses to logistic on the untouched test set, that is the result. Do not keep training until the NN "wins."

## MT5 matching without a terminal

No MT5.exe here. T1/T2/T12/T16/T35/T36 compare our code to published MT5 formulas (population std, Wilder ATR/RSI with SMA seed, iMA shift = bar offset). That is formula-match, not a live iCCI dump. Called out in the results report so nobody thinks we secretly ran the terminal.
