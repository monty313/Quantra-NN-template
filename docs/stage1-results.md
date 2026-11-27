# Stage 1 results — market-state model

Branch: `cursor/stage1-market-state-a365` on `monty313/Quantra-NN-template`.
Authoritative spec: HolyGrail Feature Specification v1.0 PDF (readable).
Why-notes for Mark: [`stage1-why-notes.md`](stage1-why-notes.md), [`stage1-plan.md`](stage1-plan.md).

Stage 1 only: calibrated P(−1/0/+1) for 15m and 1h. Not trades.

---

## Spec-level decisions (lead defaults)

| Item | Decision | Why |
|---|---|---|
| std ddof | **0** (population, divide by n) | MT5 iStdDev / iBands. Sample ddof=1 does not match. T2. |
| ATR method | **Wilder**, SMA seed of first 14 TRs then recursive | MT5 iATR. Not pandas ewm-from-bar-0 (that was adaptive-risk). T12. |
| RSI smoothing | **Wilder**, SMA seed of first `period` changes then recursive, on typical price | MT5 iRSI. T16. |
| SMA shift | **bar-index offset** after the SMA (iMA `shift`) | SMA2 period 2 shift 1 = previous bar's 2-period SMA. T35. |
| Bollinger divisor | **ddof=0**, period 5, deviation 0.5 | Same population std as iBands. T36. |
| Timezone | **naive broker-server**, do not convert to UTC | Files have no zone. Evidence against UTC below. |
| snapshot / HTF | **live_1m**, developing 5m/30m/1h | User delegated. One code path for train and live. |
| File layout | `holygrail/` package at repo root; Quantra notebooks left in place | Template is a starting folder, not the rulebook. |
| CCI MAD=0 / std=0 / ATR=0 | row **invalid**, never filled with 0 | adaptive-risk wrote CCI=0 when deviation=0. Spec forbids fill. |
| Scaler | RobustScaler 25–75 on **46 gap columns**, train only | svm-ftmo scaled all 114 columns and destroyed binaries. |
| Split | oldest 70% by 1m row order, purge label horizon, 60m embargo | TimeSeriesSplit without embargo leaks overlapping labels. |
| Rebalance | **none** | Spec. adaptive-risk used class_weight=balanced. |
| MLP | 128→64→32→3 ReLU, dropout, weight decay | Not Quantra's 5-layer sigmoid/accuracy net. |

---

## Step 0 — inspect repo + Drive

**Repo.** `Quantra-NN-template` is the Quantra “Neural Networks in Trading” course dump: Keras notebooks, `BAC.csv` daily (UTC-aware stamps, 2008–), saved `.h5` weights. Useful idea: a `create_new_model()` function. Not useful: accuracy metric, binary sigmoid, no walk-forward, no 1m MT5 path. We did not train that net.

**Drive.** Folder downloaded with gdown. Four MT5 History Center CSVs, tab-separated:

| File | Rows | Open range (naive) |
|---|---|---|
| EURUSD_M1_…2020_2026.csv | 1,998,262 | 2021-01-13 11:30 → 2026-05-27 00:00 |
| GBPUSD_M1_…csv | 1,997,455 | 2021-01-13 19:52 → 2026-05-27 00:00 |
| US30_M1_…csv | 1,998,381 | 2020-07-23 10:46 → 2026-05-26 23:59 |
| XAUUSD_M1_…csv | 1,997,634 | 2020-09-23 07:53 → 2026-05-26 22:59 |

Columns: `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>`. VOL is always 0. Seconds always 00. Timestamps are bar **opens**. Monotonic, **0 duplicate timestamps**.

**Timezone evidence (not UTC):**
- No Sunday bars on any file. UTC FX would show Sunday evening.
- EURUSD/GBPUSD: ~1440 bars/weekday, Friday last bar ~23:54–23:59, Monday first ~00:00, weekend gap ~48h.
- US30/XAUUSD: ~1380 bars/day (≈23h session + daily break).
- Friday 22:xx closes cluster around US DST weeks; most Fridays still close ~23:58 year-round. Treat as broker server time, leave naive.

**Gaps:** EURUSD 1,235 gaps >1m (weekends ~48h plus 2–7 minute holes). No interpolation.

Primary train symbol: **EURUSD** (24h, most complete). Pipeline is CSV-path generic.

---

## Step 1 — resample 1m → 5m, 30m, 1h

Left-closed, open-time labels (`pandas resample`, `closed='left'`, `label='left'`). Empty weekend bins dropped (MT5 has no bar when the market is closed).

T14: 1m open 10:04 → decision 10:05 → current 5m bar 10:00. 1m open 10:05 → decision 10:06 → current 5m bar 10:05 with one candle.

T21: 5m OHLC equals min/max/first/last of the five 1m opens in [10:00, 10:05).

T22: at decision 10:20 the 30m bar opened 10:00 uses 1m through 10:19 only.

**Exception:** T21 “match MT5's own 5m export” — Drive has 1m only, no MT5 5m files, no terminal. Pass is against the spec's left-closed rule, not a second MT5 timeframe dump.

---

## Step 2 — indicators

Applied prices as spec: CCI/Std/RSI/price-vs-SMA20 on **typical**; SMA chain, BB, ladder, sma714 on **Close**.

Developing last bar via `holygrail/asof.py`. At the last 1m of an F-bar, as-of equals the closed-bar formula (T12 ATR check).

---

## Step 3 — X, 128 columns + validity

`holygrail/columns.py` is the frozen order. `build_features()` returns float32 X, `valid` beside it, ATR_5m beside it for labels. No raw prices in X.

EURUSD: 1,998,262 rows → **1,986,221** with all 128 finite (warm-up + holes dropped, not filled).

---

## Step 4 — labels

**Training heads (CEO 2026-11-27):** `y_5m` and `y_30m` — next **closed** HTF structure, not a 15m tick.

For every 1m inside F-bar `B_t`:

- now = typical price of `B_t` once closed (shared)
- future = typical price of `B_{t+1}` (the next F-bar; not in X)
- threshold = **0.5 × ATR14 of that same F-bar series at `B_t`** (spec fraction; 5m ATR for 5m-ahead, 30m ATR for 30m-ahead)
- missing this/next F-bar → NaN, not 0

X is unchanged (128 cols, order, scaler rules, leak-safe timing). Old `y_15m` / `y_1h` are still written as unused comparison columns.

A first draft that compared `B_{t-1}` to `B_{t+1}` (skipping the bar X is watching) put class 0 back in the majority. That is not the training definition.

---

## Step 5 — split

Cutoff = timestamp of the 70% 1m row: **2024-10-11 15:14** (broker-naive). Inner val cutoff **2024-01-16 03:54**.

After purge (60m, covering next-closed 30m) + 60m embargo + both structure labels present:

| Partition | n (1m) |
|---|---|
| train (fit) | 1,104,925 |
| val (walk-forward tail) | 276,188 |
| test (untouched) | 596,634 |

T28/T29: no train row's 60m window crosses cutoff; first test row ≥ 60m after cutoff. Scaler still RobustScaler on 46 gap columns, train-fit only.

---

## Step 6 — scaler

RobustScaler, quantile 25–75, **train fit rows only**, exactly the 46 spec gap columns. Saved:

- repo `artifacts/scaler.joblib`
- repo `artifacts/scaler.json` (medians, scales, column order, cutoff, version 1.0)

---

## Step 7 — T1–T44

`python -m pytest tests/test_t1_t44.py tests/test_structure_labels.py -q` → **43 passed** (36 T1–T44 + 7 structure-label tests: shared per F-bar, missing next bar = NaN not 0, no leak after next bar closes).

Live MT5 terminal was not available. T1, T2, T12, T16, T35, T36 pass against **published MT5 formulas** (scalar vs vectorized, as-of at bar end vs closed bar). T21 passes against the spec resample rule, not an MT5 5m file.

| ID | Result | Note |
|---|---|---|
| T1 CCI vs MT5 formula | PASS | max abs err < 1e-10 |
| T2 std ddof 0 vs 1 | PASS | chose **ddof=0** |
| T3 HTF leak | PASS | future 1m mutate leaves row t unchanged |
| T4 future-change (X) | PASS | same as T3 |
| T5 scaler isolation | PASS | transform of extremes does not move center_/scale_ |
| T6 equality → 0 | PASS | |
| T7 level zones | PASS | −50 → −1, +50 → +1, 49.9 → 0 |
| T8 sign from raw | PASS | binaries unchanged after scaling |
| T9 warm-up invalid | PASS | first 100 1m invalid; last 500 valid on 220h synthetic |
| T10 zero std invalid | PASS | constant prices → all invalid, gaps NaN |
| T11 column order | PASS | |
| T12 ATR Wilder | PASS | method **wilder**; as-of at bar end matches closed |
| T13 train/live parity | PASS | prefix build last row = full build at that index |
| T14 bar boundary | PASS | |
| T15 SMA consensus | PASS | |
| T16 RSI Wilder | PASS | smoothing **wilder** |
| T17 RSI = 50 | PASS | |
| T18 RSI HTF consensus | PASS | |
| T19 col 83 | PASS | |
| T20 unused RSI isolation | PASS | definition-level (consensus inputs only) |
| T21 resample | PASS | formula/spec; no MT5 5m dump |
| T22 developing HTF | PASS | |
| T23 price vs SMA20 | PASS | |
| T24 X has 128, no y | PASS | |
| T25 label future-change | PASS | mutate after t+h leaves y_15m(t) unchanged |
| T26 threshold | PASS | |
| T27 missing label | PASS | gap → NaN, not 0 |
| T28 purge | PASS | |
| T29 embargo | PASS | |
| T30 scaler 46 cols | PASS | |
| T31 class counts | PASS | reported below |
| T32 pullback state | PASS | |
| T33 col 80 vs 83 | PASS | differ only when price blocks 80 |
| T34 prefix 1–83 | PASS | |
| T35 shifted SMA | PASS | |
| T36 Bollinger | PASS | divisor **ddof=0** |
| T37 bbs_state | PASS | |
| T38 bbs_label | PASS | |
| T39 ladder point | PASS | |
| T40 ladder bonus | PASS | p1=+1 blocks +2 |
| T41 ladder score | PASS | −3..+3 |
| T42 sma714 equal | PASS | |
| T43 shape 128 | PASS | |
| T44 scaler list | PASS | |

---

## Modeling (structure heads)

X unchanged. Fit on last 1m of each 5m bar (one snapshot per 5m; shared y). No rebalance.

**Class counts (last 1m of each 5m, no rebalance):**

| Split | Head | −1 | 0 | +1 |
|---|---|---|---|---|
| train | y_5m | 33,267 | 154,794 | 33,151 |
| train | y_30m | 30,348 | 161,491 | 29,373 |
| test | y_5m | 17,755 | 83,764 | 17,884 |
| test | y_30m | 15,734 | 87,685 | 15,984 |

Class 0 is still the mode (~70% on 5m, ~73% on 30m). Adjacent-bar typical-price changes often sit inside 0.5× that bar’s ATR. 1.0× ATR would make 0 even larger, so it was not used as the extra lever.

### Test metrics vs old 15m/1h tick heads

Old tick heads (copied from the previous run; not retrained):

| Model | Horizon | log loss | ECE | macro F1 |
|---|---|---|---|---|
| majority | 15m tick | 1.0938 | 0.0002 | 0.184 |
| logistic | 15m tick | 1.0920 | 0.0038 | 0.262 |
| LightGBM | 15m tick | 1.0909 | 0.0016 | 0.257 |
| MLP | 15m tick | 1.0913 | 0.0031 | 0.289 |
| majority | 1h tick | 1.0552 | 0.0053 | 0.190 |
| LightGBM | 1h tick | 1.0543 | 0.0081 | 0.303 |

New structure heads (this run, last 1m of each 5m):

| Model | Head | log loss | ECE | macro F1 |
|---|---|---|---|---|
| majority | y_5m | 0.8165 | 0.0018 | 0.275 |
| **logistic** | **y_5m** | **0.7410** | 0.0146 | 0.322 |
| LightGBM | y_5m | 0.7521 | 0.0037 | 0.305 |
| MLP | y_5m | 0.7482 | 0.0065 | 0.298 |
| majority | y_30m | 0.7631 | 0.0043 | 0.282 |
| logistic | y_30m | 0.7434 | 0.0131 | 0.293 |
| **LightGBM** | **y_30m** | **0.7300** | 0.0053 | 0.296 |
| MLP | y_30m | 0.7323 | 0.0179 | 0.291 |

Logloss is **not** ≈ majority: 5m logistic beats the class prior by **0.075**; 30m LightGBM by **0.033**. That is the mismatch fix. The old 15m head only beat majority by ~0.003.

Argmax still collapses to 0 (logistic 5m recall on ±1 ≈ 0.037; on 0 ≈ 0.99). Probabilities move; hard class does not. MLP does **not** beat logistic on 5m logloss; it does on 30m, barely.

**Extra lever:** skipped. Rule was “if still ≈ majority, try one more.” Logloss is not ≈ majority, so we did not run 1.0× ATR or drop-class-0. (1.0× ATR was precomputed: it makes class 0 ~93%.)

### Regime slices (LightGBM, test)

5m logloss stays in 0.746–0.762 across high/low ATR and trending/not. 30m low-ATR is easier (0.649 vs high-ATR 0.811) — quieter 30m bars are more often labeled 0, which the model already predicts.

### Explainability (val, y_5m)

SHAP mean |value| by block (LightGBM): **ladder 0.387**, bbsma 0.230, RSI 0.188, CCI 0.138, sma_chain 0.039, std 0.031, price-vs-SMA20 0.026, sma714 0.012.

Drop-one-block ablation on val LightGBM: full 0.763. Removing bbsma hurts most (+0.0042), then ladder (+0.0027). Same indicator family as X — unlike the 15m tick run, blocks actually move the loss.

**Read this as:** asking the 128-col developing state to predict the **next closed 5m/30m typical-price direction vs 0.5× same-HTF ATR** is a real (still modest) probability skill. It is not a trade signal. Stage 2 stays out of scope.

Raw JSON: `artifacts/train_results.json` (new heads). Old tick metrics: `artifacts/train_results_old_15m_1h.json`.

---

## How to rebuild

```bash
pip install -r requirements.txt
python scripts/download_mt5.py
python -m pytest tests/test_t1_t44.py tests/test_structure_labels.py -q
python scripts/build_dataset.py --csv data/mt5_raw/EURUSD_M1_202101131130_202605270000_2020_2026.csv
python scripts/train_eval.py
```

Policy: `fit(X, y_5m, y_30m)` / `predict_proba(X)` → dict of (n, 3) columns [−1, 0, +1] / `explain(X)`.
