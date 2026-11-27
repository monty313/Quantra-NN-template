# Stage 1 — why the code looks like this

Mark: read this next to the repo. Short reasons, not a second spec.

## Three layers, on purpose

1. `holygrail/` builds X and validity from 1m candles. Same function for history and live.
2. `holygrail/labels.py` builds answers in a **separate** table. Targets never enter X or the scaler. Training heads are next-closed **5m** and **30m** structure. Old y_15m / y_1h stay as unused comparison columns.
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

X is set 2: live 1m + developing 5m/30m/1h **structure**. A 15-minute 1m-tick vs 0.5×5m ATR asks “did the next 15 ones-and-zeros of price clear a 5m ATR band?” That is not the question the 128 columns describe, which is why those heads sat on the majority.

**Training heads (CEO 2026-11-27):** next **closed** 5m and next **closed** 30m structure direction.

- `now` = typical price of **this** F-bar when it has closed (B_t). Shared for every 1m inside that bar. Labels may use that close; early-row X does not.
- `future` = typical price of the **next** F-bar (B_{t+1}). X never contains B_{t+1}. We do not skip B_t (that compared B_{t-1} to B_{t+1} and put class 0 back in the majority).
- threshold = **0.5 × ATR14 of that same F-bar series**, at B_t. Spec already names 0.5× ATR. 5m-ahead uses 5m ATR; 30m-ahead uses 30m ATR.
- Missing this/next F-bar (weekend, tail) → NaN, not 0. Same rule as T27.

Old y_15m / y_1h are still computed and written, but they are not fit.

Train on the last 1m of each 5m bar (one live snapshot per 5m, most developed X, still leak-free for B_{t+1}).

If both heads still ≈ majority, one extra lever: **1.0×** the same HTF ATR (wider, less class-0 mush). Not a second architecture. Stop after that.

## Split

Oldest 70% of 1m history by row order. Purge train rows whose 1h window crosses the cutoff. First test row ≥ 60 minutes after cutoff. Val is the time-ordered tail of train with the same purge/embargo. Never shuffled.

`svm-ftmo` used TimeSeriesSplit (good) but no embargo (bad for overlapping 15m labels).

No class_weight. Spec: report counts, do not rebalance. `adaptive-risk` balanced classes — that makes logloss and calibration lie.

## Overlap

Consecutive 1m rows of the same 5m bar share one y_5m (and 30m rows share y_30m). We train on the last 1m of each 5m bar, not on 1.1M independent samples.

## Models

- Majority: constant train class frequencies (fair logloss baseline).
- Logistic: L2 multinomial, no rebalance.
- LightGBM: shallow, early stop on time-ordered val.
- MLP: 128→64→32→3, dropout, weight decay, **not** Quantra's 5-layer sigmoid tower.

If MLP loses to logistic on the untouched test set, that is the result. Do not keep training until the NN "wins." `v6_learn` LESSONS_BOOK called that "stamping the day": the score moves, the skill does not.

Prior-repo map (what we refused vs kept): `docs/stage1-plan.md` § Lessons.

## MT5 matching without a terminal

No MT5.exe here. T1/T2/T12/T16/T35/T36 compare our code to published MT5 formulas (population std, Wilder ATR/RSI with SMA seed, iMA shift = bar offset). That is formula-match, not a live iCCI dump. Called out in the results report so nobody thinks we secretly ran the terminal.
