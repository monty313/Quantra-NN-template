# Round 3 — y_5m_nextclose_vs_typ, logistic 180 / 128, fold_60_20

New label only. 180-col builder unchanged. `y_5m_dir` kept as reference, not deleted.
Sealed newest 20% unscored. No LightGBM/MLP. No new features. PR #1 untouched.

JSON: `/cursor/stores/bc-63d4488e-eaab-49e7-b822-99139192b098/docs/round3-label-typ.json`

## Label

After this 5m bar closes, every 5 minutes: **1** if close of the **next** 5m > typical of **this** 5m, **0** if lower.
Typical = (H+L+C)/3 of this bar. Direction only. Ties (next close == this typical) dropped, counted.
Labels are never features. X uses data up to this bar’s close only.

- fold: `fold_60_20`
- train n after purge + finite new label: **196,398**
- val n finite new label: **65,394**
- ties dropped train / val: **982** / **403**
- sealed rows in this parquet: 65,797 (not scored)

## Val accuracy

| Model / baseline | n | Acc |
|---|---|---|
| **Logistic 180** (primary) | 65,394 | **61.75%** |
| Logistic 128 | 65,394 | 61.71% |
| Rule: this close > this typical else 0 | 65,394 | **61.90%** |
| Majority (train mode = 1) | 65,394 | 50.78% |
| prev5m_dir (+1→1, −1→0; zeros dropped) | 64,921 | 57.27% |

**KEY: logistic 180 minus rule = -0.15 pp.**

Logistic 180 logloss 0.6553. Logistic 128 logloss 0.6558.

## When the 180-col model disagrees with the rule

- Agree n=61,622 acc=62.55%
- Disagree n=3,772 acc=48.70%
- Model vs rule agreement rate: 94.23%

## 10-bin calibration (equal-count, P(y=1), logistic 180)

| Bin | n | Mean pred | Actual rate |
|---|---|---|---|
| 1 | 6,539 | 0.2870 | 0.2841 |
| 2 | 6,539 | 0.3640 | 0.3510 |
| 3 | 6,540 | 0.4078 | 0.3890 |
| 4 | 6,539 | 0.4472 | 0.4424 |
| 5 | 6,540 | 0.4854 | 0.4852 |
| 6 | 6,539 | 0.5235 | 0.5278 |
| 7 | 6,539 | 0.5611 | 0.5828 |
| 8 | 6,540 | 0.5997 | 0.6269 |
| 9 | 6,539 | 0.6429 | 0.6623 |
| 10 | 6,540 | 0.7192 | 0.7268 |

## Acc by hour (logistic 180, val last_1m hour)

| Hour | n | Acc |
|---|---|---|
| 00 | 1,957 | 62.03% |
| 01 | 2,323 | 61.17% |
| 02 | 2,476 | 61.35% |
| 03 | 2,595 | 60.81% |
| 04 | 2,564 | 62.64% |
| 05 | 2,555 | 62.23% |
| 06 | 2,544 | 62.22% |
| 07 | 2,497 | 62.23% |
| 08 | 2,565 | 63.16% |
| 09 | 2,594 | 59.87% |
| 10 | 2,607 | 61.91% |
| 11 | 2,587 | 61.58% |
| 12 | 2,580 | 60.89% |
| 13 | 2,567 | 60.77% |
| 14 | 2,572 | 61.78% |
| 15 | 2,559 | 61.86% |
| 16 | 2,839 | 61.82% |
| 17 | 3,278 | 62.17% |
| 18 | 3,268 | 61.23% |
| 19 | 3,217 | 62.11% |
| 20 | 3,238 | 63.22% |
| 21 | 3,252 | 62.79% |
| 22 | 3,217 | 61.55% |
| 23 | 2,943 | 60.24% |

## Acc by weekday (Mon=0 … Fri=4)

| Weekday | n | Acc |
|---|---|---|
| Mon | 4,818 | 62.35% |
| Tue | 15,063 | 62.03% |
| Wed | 14,914 | 61.37% |
| Thu | 15,231 | 62.04% |
| Fri | 15,368 | 61.37% |

## Top 10 logistic 180 coefficients by |coef| (scaled space)

| Rank | Feature | Coef | Sign |
|---|---|---|---|
| 1 | `ltf_price_sma20_gap_5m` | -3.8224 | − (y=1) |
| 2 | `bbs_sma20_gap_5m` | +1.6697 | + (y=1) |
| 3 | `s20_gap_5m` | +1.6697 | + (y=1) |
| 4 | `lad_g1_5m` | -0.5399 | − (y=1) |
| 5 | `bbs_mid_gap_5m` | +0.5331 | + (y=1) |
| 6 | `bbs_mid_gap_30m` | -0.4453 | − (y=1) |
| 7 | `rsi14_dist50_5m` | -0.3961 | − (y=1) |
| 8 | `sma_bc_gap_5m` | +0.2869 | + (y=1) |
| 9 | `lad_g2_5m` | -0.2799 | − (y=1) |
| 10 | `lad_50gap_5m` | +0.2637 | + (y=1) |

## Placebo (frozen logistic 180 preds vs shuffled val y)

- Shuffled-label (permute val y, seed 0): n=65,394 acc=49.94%
- Shuffled-time (1h blocks of 12 5m bars): n=65,394 acc=49.91%

## Notes

- 0/1 label mapped to −1/+1 only at `LogisticPolicy.fit` (`encode_y` rejects class 0).
- P(y=1) = proba column for class +1.
- Rule baseline uses this bar’s close vs this bar’s typical (known at t). Else 0.
- `y_5m_dir` still in parquet. Not the train label.
- Agreement new-label vs mapped `y_5m_dir` on val (both finite): 91.48% n=65,394.
