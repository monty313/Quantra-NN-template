# Phase C — val-fold models (stop here)

**Phase C only. VAL ONLY. Sealed test was not scored. No Phase D. No test tuning.**

Branch: `cursor/stage1-market-state-a365`. Replay: `python scripts/phase_c_models.py` → `artifacts/phase_c_val.json`.
`y_5m_dir` unchanged (alias of `y_5m`). Same 70/30 cutoff + 120m purge + 60m embargo. RobustScaler fit on **train 67 gaps only**.

---

## Rebuild

EURUSD 1m → 180-col X. FEATURE_VERSION `1.1`. Scaler 67 columns, `n_train` 1,087,104.

| Split clock | Value |
|---|---|
| cutoff | 2024-10-11 15:14 |
| inner_cutoff | 2024-01-15 09:14 |
| train 1m / val 1m | 1,087,104 / 271,740 |
| train last-1m-of-5m / val last-1m-of-5m | **217,642 / 54,369** |
| train unique 5m bars | 217,644 |

Same cutoffs as the 128-col run. Valid-row counts did not move. Test parquet was written with the train-fit scaler and **never opened for metrics**.

---

## Sampling and effective N

Val metrics always use **last 1m of each 5m** (one row per label). Two train schemes:

| Scheme | Train rows | Kish n_eff | Unique 5m bars |
|---|---|---|---|
| **thin** last-1m-of-5m | 217,642 | **217,642** | 217,642 |
| **bar weights** all 1m, `w = 1/n_1m_in_bar` | 1,087,104 | 1,086,010 | 217,644 |

Kish on nearly-equal 1/5 weights ≈ n_1m, not n_bars. The clustered N is still ~217k bars. Thin is the honest N for a shared-per-bar label.

**Thin wins.** Weighted logistic 180: val logloss **0.6351** vs thin **0.6286**, ECE 0.047 vs 0.015. Weighted LGBM worse still. Intra-bar 1m copies are not extra independent observations.

---

## Trial log (val last-1m, `y_5m_dir`)

n_val = 54,369. Majority class on val is +1 (50.86%).

| ID | Model | X | Train | logloss | acc | vs 50% | ECE | Notes |
|---|---|---|---|---|---|---|---|---|
| C1 | majority | — | thin | 0.6931 | 50.86% | +0.86 | 0.0050 | class prior |
| C2 | persistence | `prev5m_dir` | thin | 0.6763 | 59.16% | +9.16 | 0.0094 | train P(continue)=58.30% |
| C3 | logistic | 128 | thin | 0.6293 | 64.94% | +14.94 | 0.0154 | |
| **C4** | **logistic** | **180** | **thin** | **0.6286** | **64.97%** | **+14.97** | **0.0147** | **winner** |
| C5 | LightGBM | 128 | thin | 0.6387 | 63.68% | +13.68 | 0.0172 | bagging_freq was 0; 400/400 trees |
| C6 | LightGBM | 180 | thin | 0.6354 | 64.11% | +14.11 | 0.0183 | same |
| C7 | logistic | 180 | weights | 0.6351 | 64.92% | +14.92 | 0.0473 | worse logloss + ECE |
| C8 | LightGBM | 180 | weights | 0.6458 | 63.85% | +13.85 | 0.0566 | |
| C9 | MLP 180→64→32→2 | 180 | thin | 0.6332 | 64.37% | +14.37 | 0.0090 | loses to logistic |
| C10 | LightGBM | 128 | thin | 0.6359 | 63.98% | +13.98 | — | bagging on, 800 trees, best_iter 797 |
| C11 | LightGBM | 180 | thin | 0.6332 | 64.36% | +14.36 | — | bagging on, best_iter 799 |

LightGBM params (C5–C11): depth 4, leaves 16, lr 0.03, min_child 200, feature_fraction 0.8, L2=2. C10/C11 also bagging 0.8 freq 1. Early stop on this val fold. Trees hit the cap — more trees were not going to beat logistic 0.6286.

MLP on the winning feature set (180). Does not beat logistic. Better ECE, worse logloss/acc.

**180 vs 128 logistic: −0.0006 logloss, +0.03 pp accuracy.** Phase B columns are almost free on this label.

---

## Ablation (drop whole block, retrain logistic 180, val)

Positive delta = block was useful (logloss got worse when dropped).

| Block | Δ logloss vs full 0.6286 |
|---|---|
| **price_vs_sma20** | **+0.00508** |
| sma4hl_rsi_bb (block 10) | +0.00054 |
| sma_chain | +0.00018 |
| bbsma | +0.00009 |
| ladder | +0.00006 |
| rsi_pullback | +0.00005 |
| rsisma_trio (block 8) | +0.00004 |
| cci | +0.00002 |
| prev5m_dir (block 11) | −0.00002 |
| std | −0.00002 |
| sma714 | −0.00007 |
| rsi_bb_reversion (block 9) | −0.00012 |

The 5m typical-vs-SMA20 gap is still the load-bearing column. New RSI-BB block is noise-to-negative. SMA4 high/low gaps help a little. `prev5m_dir` is already in the persistence baseline; the linear model barely uses it on top of the other state.

---

## SHAP / coefficients

Tree SHAP on LGBM 180 (C6, val sample 2000). Top columns: `lad_g1_5m`, `rsi3_dist50_5m`, **`s4_lo_gap_5m`**, **`s4_hi_gap_5m`**, `lad_g3_5m`. Block sums: ladder 0.495, sma4hl 0.374, rsi_pullback 0.289, cci 0.179, rsisma 0.089, prev5m_dir 0.034.

Logistic |coef| (180, thin): `ltf_price_sma20_gap_5m` **5.04**, then `bbs_sma20_gap_5m` = `s20_gap_5m` **2.05** (same Close−SMA20/ATR, duplicated). `prev5m_dir` coef ≈ 0. Block |coef| sums: price_vs_sma20 5.05, bbsma 3.72, sma4hl 2.99, ladder 2.31, rsi_pullback 2.06.

Trees like ladder + SMA4 channel. Linear model likes the 5m SMA20 gap. Neither reads `B_{t+1}`.

---

## Mentor note — 95% is still blocked

Val logistic 180 is **64.97%**. Persistence is **59.16%**. Majority **50.86%**.

Phase A on the **sealed test** (not used here): flip rate **41.36%**, persist ceiling **58.64%**, logistic 128 **64.87%**. To hit 95% you still need ~88% of those flips right. The flips live in `B_{t+1}`. Ablation and SHAP say the new columns describe *this* bar’s RSI/SMA/BB state. They added **0.03 pp**.

Do not plan on 95%. A honest next question is whether Phase D’s sealed test agrees that 180 ≈ 128, not whether 180 suddenly sees the next bar.

---

## What this turn did not do

- No Phase D.
- No test metrics, no test plots, no test SHAP.
- No extra LGBM grid (depth 3 vs 5, leaves 8 vs 31). One regularized setting, then bagging-on confirm. Logistic still wins.
- Did not retrain on test. Did not change `y_5m_dir`.
