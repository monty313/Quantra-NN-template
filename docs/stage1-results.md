# Stage 1 results — market-state model

Branch: `cursor/stage1-market-state-a365` on `monty313/Quantra-NN-template`.
Authoritative spec: HolyGrail Feature Specification v1.0 PDF (readable).
Why-notes for Mark: [`stage1-why-notes.md`](stage1-why-notes.md), [`stage1-plan.md`](stage1-plan.md).

Stage 1 only: calibrated P(−1/+1) for next-closed **5m / 30m / 1h** direction. Not trades. ATR size fence was the wrong target.

**Phase B** appended columns 129–180 (RSI/SMA/BB + `prev5m_dir`). Code and tests are 180-col / RobustScaler 67. **This results file and `artifacts/train_results.json` are still the 128-col run.** No rebuild, no train. See [`phase-b-features.md`](phase-b-features.md).

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
| Split | oldest 70% by 1m row order, **120m purge** (1h remainder + next 1h), 60m embargo | TimeSeriesSplit without embargo leaks overlapping labels. |
| Rebalance | **none** | Spec. adaptive-risk used class_weight=balanced. |
| MLP | 128→64→32→2 ReLU, dropout, weight decay | Not Quantra's 5-layer sigmoid/accuracy net. |

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

**Training heads (direction only):** `y_5m`, `y_30m`, and `y_1h_structure`.

For every 1m inside F-bar `B_t` of that head’s timeframe:

- now = typical price of `B_t` once closed (shared)
- future = typical price of `B_{t+1}` (the next F-bar; not in X)
- **y = sign(future − now) ∈ {−1, +1}**
- **no 0.5× ATR fence.** Std / ATR / RSI stay in X only.
- equal typical (exact 0) → invalid, **drop**, never class 0
- shared across every 1m inside the F-bar
- missing this/next F-bar → NaN, not 0

X is unchanged (128 cols, order, scaler rules, leak-safe timing). Old `y_15m` / `y_1h` tick heads are still written as unused comparison columns.

The ATR size fence was the wrong target: it manufactured a ~70% class-0 majority and made argmax collapse. Direction-only is the question the developing HTF columns actually describe.

---

## Step 5 — split

Cutoff = timestamp of the 70% 1m row: **2024-10-11 15:14** (broker-naive). Inner val cutoff **2024-01-15 09:14**.

After purge (**120m**, covering the remainder of the current 1h plus the next closed 1h) + 60m embargo + all three direction labels present (exact-zero TP moves dropped):

| Partition | n (1m) |
|---|---|
| train (fit) | 1,087,104 |
| val (walk-forward tail) | 271,740 |
| test (untouched) | 587,618 |

Embargo is still 60m (T29). Purge is 120m so a 1h label whose future bar straddles the cutoff cannot sit in train. Scaler still RobustScaler on 46 gap columns, train-fit only.

---

## Step 6 — scaler

RobustScaler, quantile 25–75, **train fit rows only**, exactly the 46 spec gap columns. Saved:

- repo `artifacts/scaler.joblib`
- repo `artifacts/scaler.json` (medians, scales, column order, cutoff, version 1.0)

---

## Step 7 — T1–T44

`python -m pytest tests/test_t1_t44.py tests/test_structure_labels.py -q` → **46 passed** (36 T1–T44 + 10 structure-label tests: shared per 5m/30m/1h F-bar, missing next bar = NaN not 0, equal typical = invalid not class 0, y ∈ {−1,+1} only).

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

## Modeling (2-class direction heads)

X frozen. Fit on last 1m of each 5m bar (one snapshot per 5m; y_5m unique, y_30m repeats ~6×, y_1h_structure repeats ~12×). No rebalance. Policy is `MultiHead` with keys `y_5m` / `y_30m` / `y_1h_structure`. `predict_proba` columns are **[−1, +1]**.

**Class counts (last 1m of each 5m, no rebalance, no class 0):**

| Split | Head | −1 | +1 |
|---|---|---|---|
| train | y_5m | 108,046 | 109,596 |
| train | y_30m | 108,489 | 109,153 |
| train | y_1h_structure | 109,385 | 108,257 |
| test | y_5m | 58,532 | 59,066 |
| test | y_30m | 58,470 | 59,128 |
| test | y_1h_structure | 58,529 | 59,069 |

Almost 50/50. Exact-zero typical moves were dropped, not labeled 0.

### Test metrics (EURUSD, last 1m of each 5m)

Majority logloss is ~ln(2) = 0.693, as a fair coin. Accuracy vs 50% is the hard-class check.

| Model | Head | log loss | acc | acc−50% | ECE | −1 P/R | +1 P/R |
|---|---|---|---|---|---|---|---|
| majority | y_5m | 0.6931 | 0.5023 | +0.0023 | 0.0013 | 0.000/0.000 | 0.502/1.000 |
| **logistic** | **y_5m** | **0.6292** | **0.6487** | **+0.1487** | 0.0142 | 0.650/0.638 | 0.648/0.659 |
| LightGBM | y_5m | 0.6408 | 0.6322 | +0.1322 | 0.0136 | 0.634/0.618 | 0.631/0.647 |
| MLP | y_5m | 0.6359 | 0.6395 | +0.1395 | 0.0048 | 0.649/0.602 | 0.632/0.677 |
| majority | y_30m | 0.6931 | 0.5028 | +0.0028 | 0.0013 | 0.000/0.000 | 0.503/1.000 |
| **logistic** | **y_30m** | **0.6757** | **0.5733** | **+0.0733** | 0.0089 | 0.572/0.567 | 0.575/0.580 |
| LightGBM | y_30m | 0.6769 | 0.5714 | +0.0714 | 0.0014 | 0.568/0.575 | 0.575/0.567 |
| MLP | y_30m | 0.6765 | 0.5690 | +0.0690 | 0.0078 | 0.566/0.573 | 0.572/0.565 |
| majority | y_1h_structure | 0.6932 | 0.4977 | −0.0023 | 0.0049 | 0.498/1.000 | 0.000/0.000 |
| **logistic** | **y_1h_structure** | **0.6741** | 0.5746 | +0.0746 | 0.0071 | 0.570/0.592 | 0.580/0.558 |
| LightGBM | y_1h_structure | 0.6772 | 0.5707 | +0.0707 | 0.0125 | 0.568/0.575 | 0.573/0.567 |
| MLP | y_1h_structure | 0.6756 | **0.5757** | **+0.0757** | 0.0016 | 0.569/0.605 | 0.583/0.546 |

Logistic is best logloss on every head (MLP does not beat it). Hard class now works: 5m logistic **64.9% vs 50%**; 30m/1h ~57.5% vs 50%. Per-class recall is balanced — the ATR fence's "always predict 0" failure is gone.

Old 15m/1h tick heads (not retrained): majority 15m logloss 1.0938, best LGBM 1.0909 (Δ −0.003). Kept as unused comparison.

### Regime slices (LightGBM, test)

5m acc stays in 0.624–0.640 across high/low ATR and trending/not (high-ATR slightly easier: 0.640 vs 0.624). 30m/1h sit in 0.561–0.577 — quieter vs noisier does not recreate a class-0 mush.

### Explainability (val, y_5m)

SHAP mean |value| by block (LightGBM): **ladder 0.562**, bbsma 0.308, RSI 0.294, CCI 0.160, price-vs-SMA20 0.052, sma_chain 0.020, std 0.018, sma714 0.004.

Drop-one-block ablation on val LightGBM: full 0.6395. Removing bbsma hurts most (+0.0031), then ladder (+0.0009). Same indicator family as X.

**Read this as:** asking the 128-col developing state to predict the **sign of the next closed 5m / 30m / 1h typical-price change** is a real probability *and* hard-class skill. Strongest on 5m. It is not a trade signal. Stage 2 stays out of scope.

Raw JSON: `artifacts/train_results.json`. Old tick metrics: `artifacts/train_results_old_15m_1h.json`.

---

## How to rebuild

```bash
pip install -r requirements.txt
python scripts/download_mt5.py
python -m pytest tests/test_t1_t44.py tests/test_structure_labels.py -q
python scripts/build_dataset.py --csv data/mt5_raw/EURUSD_M1_202101131130_202605270000_2020_2026.csv
python scripts/train_eval.py
```

Policy: `fit(X, ys)` with `ys` keyed `y_5m` / `y_30m` / `y_1h_structure` / `predict_proba(X)` → dict of (n, 2) columns [−1, +1] / `explain(X)`.
