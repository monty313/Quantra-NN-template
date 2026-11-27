# Stage 1 market-state model — plan

Status: done on `cursor/stage1-market-state-a365`. T1–T44 + structure-label tests green (45). Training heads are next-closed **5m / 30m / 1h** structure (`y_15m`/`y_1h` unused comparison). EURUSD train/eval in `artifacts/train_results.json` and `docs/stage1-results.md`.

## Goal

Learn market structure from a 128-column observation matrix `X` plus three 3-class **structure** targets (`y_5m`, `y_30m`, `y_1h_structure`: next closed HTF direction). Old `y_15m` / `y_1h` tick heads are unused comparison columns. Outputs are calibrated state probabilities, not trades.

## Layers

1. Feature pipeline (`holygrail/`): 1m MT5 → developing 5m/30m/1h → 128-col `X` + validity.
2. Labels: `y_5m` / `y_30m` / `y_1h_structure` training heads stored separately; `y_15m` / `y_1h` kept for comparison.
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

Studied the monty313 owner list below (skipped med-reminder). Goal: steal the good habits, not the patterns that made earlier bots look good and then fail. Stage 1 is HolyGrail 128-col observation + calibrated 3-class probabilities. It is not Camillion 114, not regimes StdDev 20/30, not PPO, not FTMO PnL.

### Keep

- Chronological split + **purge of the label horizon** (`adaptive-risk-ml-trader` `train.py`). Stage 1 also adds a **60-minute embargo**, which that repo mostly skipped.
- Causality tests: mutating a future HTF bar must not change earlier rows. Same idea as our T3/T4/T25.
- Never put raw prices in the model (`MARK_TRADING_MIND` invariance). Prices stay inside indicators/labels only.
- Observe-only until a later stage (`quant-reasoning`, `HG-Quote-Trend`). Stage 1 outputs probabilities, not orders.
- Do not invent missing indicator settings (`quant-reasoning`). Timezone stays naive broker-server; MT5 methods are named and tested.
- Drop incomplete rows. `svm-ftmo` at least used `dropna`. We go further: **flag invalid, never fill**.
- Honest failed tests stay failed (`MARK_TRADING_MIND` Kronos entropy vs next-15-bar move). Do not promote a failed extra indicator into X.
- One formula source, tested (`rl-trading-live` DESIGN: talib vs numpy must not mix in a parity run). We pick MT5 formulas and test them.

### Do not copy

- Quantra accuracy / binary sigmoid / daily BAC (`Quantra-NN-template`, `Q-Trading-with-Machine-Learning-Regression`).
- `class_weight="balanced"` (`adaptive-risk-ml-trader`, `DECISON-TREE`, copies in `MSM-strategies`).
- Median impute / `fillna(0)` (`adaptive-risk` SimpleImputer).
- StandardScaler on **every** column (`svm-ftmo` 114 Camillion features).
- `TimeSeriesSplit` with **no embargo** (`svm-ftmo`, Quantra regression notebooks).
- HTF **ffill onto LTF** (`svm-ftmo` `features/indicators.py`). That pastes a finished HTF close into earlier 5m snapshots.
- Forming-bar peek-ahead (Fallout `build.py`, called out in the SVM course-repo handoff).
- CCI=0 when std=0; ATR as pandas `ewm` from bar 0 with no SMA seed.
- Empty / auto README shells that say `python main.py` when there is no `main.py`.
- Mixing HolyGrail 128-col with `regimes` StdDev 20/30, Camillion 114, or 367-dim RL obs.
- Kronos entropy as a feature after it failed.
- Placeholder nets with random weights shipped as "the model".
- PnL / FTMO / lot size / forced-entry / live MT5 in Stage 1.
- Training until the NN "wins" vs logistic. If the gap is 0.0007 logloss, say so.

### Per-repo (priority list)

| Repo | What it actually is | Mistake we refuse | Habit we keep |
|---|---|---|---|
| **Quantra-NN-template** (this repo, course dump) | Keras notebooks, `BAC.csv` daily, accuracy + sigmoid | Lock Stage 1 to that net / metric | `create_new_model()` idea only; folder is a starting point |
| **adaptive-risk-ml-trader** | Daily-target ML research, walk-forward | `class_weight=balanced`, median impute, ewm ATR, CCI fill 0 | Time split + purge; log_loss in the report |
| **quant-reasoning** | Observation-only harness | Invent timezone or orders | OBSERVE_ONLY; do not guess MISSING settings |
| **nn-ftmo-trading-bot** | README shell, no code | Empty template as if it were a bot | Real scripts + tests in this PR |
| **regimes** | StdDev 20/30 + ATR 4/14 **definitions** | Mix those states into 128-col X | Definitions stay a later overlay, not features |
| **LSTM-Gold-Momentum-Hunter** | README shell | Pretend there is an LSTM gold hunter to port | — |
| **svm-ftmo-trading-bot** | Camillion 114 + SVM + FTMO bot | Scale all cols; ffill HTF; no embargo; quantile future-return labels | No shuffle; dropna; chronological cut |
| **HG-Quote-Trend** | thinkorswim Quotes / scans, not a model | Rebuild HTF from LTF; invent math; TOS keep-last NaN as a fill rule | Never invent math; observe-only; real bars |
| **MARK_TRADING_MIND** | Fused-brain plan + Kronos probe | Entropy-as-feature; random-weight bridge as a model | Returns/ranks not raw prices; publish the fail |
| **MARK-TRADING-UNIFIED** | Graph / docs / auto README (`python main.py` missing) | Copy the shell README | One real run path documented |
| **Camillion_Claude_RL_model** | 367-dim RL, 64 alphas, reward retunes, FTMO | Mix Camillion stack; train on PnL | Leave RL for Stage 2+ |
| **Qunatra_Deep-Reinforcement-Learning-in-Trading** | Spec-only PPO blueprint, no code | Implement PPO from the markdown | — |
| **Q-Trading-with-Machine-Learning-Regression** | Quantra: next-day GLD high/low | Daily regression target; StandardScaler; TimeSeriesSplit as enough | Course is reference, not the pipeline |
| **-Trading-with-Machine-Learning-Classification-and-SVM** | Course + `SVM.py` that **places orders** | Run order code; forming-bar peek-ahead | Plan's forming-bars + purge/embargo (we implemented those) |
| **AI-Trading-Floor** | README + UI_PLAN only | Copy a floor that has no model | — |
| **the-truth** | RL scalper lineages, PnL, class_weight | Stage 1 as a live bot | — |
| **vizual_nn** | Viz lab; README says `main.py`, entry is `run.py` | Fake quickstart | Honest README |
| **MT5-RL** | Empty (`.gitattributes` only) | — | — |
| **rl-trading-live** | PPO: direction + lot + exits; forced entry; 2.5%/1% DD reward | Policy that sizes and exits; mix talib and numpy | One indicator formula, tested |
| **deep-reinforcement-learning-trading** | Abandoned DQN, superseded | Port `gpu_rl_trading/` | — |
| **meta-learning-rl-model-8-21-2026** | US30 learner, locks, cameras | Reward shaping / "stamp the day"; cameras as badges | Cameras as **observation**, not a score |
| **final-rl-model-6_13** / **RL-model-trading-bot-ppo-mlp_Claude-** | Duplicate Quantra cockpit + barbershop PPO | Duplicate RL stacks into Stage 1 | — |
| **v6_learn** | LESSONS_BOOK: don't stamp the day; weights overwritten | Keep-training-until-NN-wins; mix products | Honest metrics; freeze artifacts |
| **day-trading-gcp** | Method + Terraform, not the learner | Mix infra / GCP into the feature matrix | — |
| **ARMY** | Ops / lab dump of other bots | Pull mixed stacks from the kitchen sink | — |
| **MSM-strategies** | Curriculum copies of DECISON-TREE / svm-ftmo / regimes | Treat the copy as a new method | Same mistakes, don't re-import |
| **DECISON-TREE** | sklearn/CART, `class_weight`, 70/15/15 **index slice**, FTMO backtest | class_weight; split without embargo; tree PnL as Stage 1 score | Freeze a fitted model to a file; time-ordered cut |

**Why this matters for Mark reading the code:** every surprising choice (ddof=0, Wilder seed, no fill, 70/30 by row time, scaler only on gaps, no class_weight) is there to avoid a mistake that already showed up in one of these repos.
