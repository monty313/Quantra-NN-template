# Phase D — one sealed test pass (stop here)

**Phase D only. One test look. No tuning. No threshold search on test. `y_5m_dir` unchanged.**

Winner from C, frozen: **logistic 180**, thin last-1m-of-5m, RobustScaler train 67 gaps. Same split (cutoff 2024-10-11 15:14, inner 2024-01-15 09:14).

Replay: `python scripts/phase_d_test.py` → `artifacts/phase_d_test.json`.

---

## Trial log

| ID | Step | Split | n | logloss | acc | vs 50% | Notes |
|---|---|---|---|---|---|---|---|
| D1 | logistic 180 reliability | **val** | 54,369 | 0.6286 | 64.97% | +14.97 | 10 equal-count bins; ECE_q **0.0159**; Cox slope **1.078**, intercept **0.019** |
| D1h | high conf (bins 1∪10) | val | 10,874 (20%) | 0.5199 | **78.48%** | +28.48 | VAL-chosen outer deciles |
| D1l | low conf (bins 5–6) | val | 10,873 (20%) | 0.6921 | 51.87% | +1.87 | coin |
| **D2d** | **logistic 180 freeze** | **test** | **117,598** | **0.6289** | **64.87%** | **+14.87** | **one pass** |
| D2c | logistic 128 | test | 117,598 | 0.6292 | 64.87% | +14.87 | frozen comparison, not a pick |
| D2a | majority | test | 117,598 | 0.6931 | 50.23% | +0.23 | same rows |
| D2b | persistence `prev5m_dir` | test | 117,598 | 0.6783 | 58.56% | +8.56 | train P(continue)=58.30% |
| D2h | high conf, **VAL edges** | test | 23,182 (19.7%) | 0.5110 | **78.92%** | +28.92 | not retuned on test |
| D2l | low conf, VAL edges | test | 23,511 (20.0%) | — | 52.93% | +2.93 | |
| D3a | placebo t+2, frozen | test | consecutive next bar | — | **49.01%** | −0.99 | **PASS ≤52%** |
| D3b | 1h block-shuffle y, frozen | test | 117,598 | — | **50.05%** | +0.05 | **PASS ≤52%** |

D3 **go**.

---

## D1 — reliability on VAL only

10 equal-count bins of `P(+1)` from the frozen logistic 180 (fit on train thin; val scores only).

| Bin | P(+1) range | n | mean p | emp +1 | \|gap\| | hard acc |
|---|---|---|---|---|---|---|
| 1 | 0.001–0.287 | 5437 | 0.226 | 0.212 | 0.014 | 78.8% |
| 2 | 0.287–0.352 | 5437 | 0.321 | 0.306 | 0.015 | 69.4% |
| 3 | 0.352–0.405 | 5437 | 0.379 | 0.358 | 0.020 | 64.2% |
| 4 | 0.405–0.456 | 5437 | 0.430 | 0.422 | 0.008 | 57.8% |
| 5 | 0.456–0.505 | 5436 | 0.481 | 0.496 | 0.015 | 50.4% |
| 6 | 0.505–0.555 | 5437 | 0.530 | 0.533 | 0.003 | 53.3% |
| 7 | 0.555–0.604 | 5437 | 0.579 | 0.591 | 0.012 | 59.1% |
| 8 | 0.604–0.656 | 5437 | 0.630 | 0.669 | 0.040 | 66.9% |
| 9 | 0.656–0.720 | 5437 | 0.686 | 0.716 | 0.030 | 71.6% |
| 10 | 0.720–0.997 | 5437 | 0.780 | 0.782 | 0.002 | 78.2% |

- Quantile reliability ECE: **0.0159**
- Equal-width confidence ECE (existing metric): **0.0147**
- Bin OLS (emp ~ a + b·mean_p): slope **1.081**, intercept **−0.036**
- Cox: logit P(y=+1) = **0.019 + 1.078** logit(p). Near 0/1. Slightly underconfident in the upper-middle bins (8–9).

**Frozen bands (not touched after this):**

- High-confidence: outer deciles, `P(+1) ≤ 0.287` or `P(+1) ≥ 0.720`
- Low-confidence: bins 5–6, `0.456 < P(+1) ≤ 0.555`

No threshold was chosen on test.

---

## D2 — freeze, score TEST once

n_test last-1m-of-5m = **117,598**. Same row set as the old 128-col 5m logistic run.

| Model | logloss | acc | ECE |
|---|---|---|---|
| majority | 0.6931 | 50.23% | 0.0013 |
| persistence | 0.6783 | 58.56% | 0.0031 |
| logistic 128 | 0.6292 | 64.87% | 0.0142 |
| **logistic 180** | **0.6289** | **64.87%** | **0.0128** |

180 vs 128 on the sealed test: **same 64.87% accuracy**, logloss −0.0003. Phase B columns did not move the number that was already on the board.

Per-class 180: −1 P/R 0.650/0.637, +1 P/R 0.647/0.661. Balanced. Not a class-0 collapse.

High-confidence slice using **VAL edges only**: n=23,182, acc **78.92%**, logloss 0.511. Low-confidence: n=23,511, acc 52.93%. The model’s probabilities rank. They do not turn the middle into 95%.

Test Cox slope 1.087, intercept −0.014. Quantile ECE on test with **val edges**: 0.0132.

---

## D3 — leak controls (frozen model)

No refit. Frozen P(X) vs a broken label.

| Check | Acc | Pass ≤52% |
|---|---|---|
| Placebo: P(X at end of B_t) vs `y_5m_dir[t+1]` = sign(TP(B_{t+2})−TP(B_{t+1})) | **49.01%** | **PASS** |
| 1h block-shuffle: permute test y in 12 consecutive 5m rows | **50.05%** | **PASS** |

The frozen 64.9% skill does not transfer to the next bar’s direction and does not survive a 1h label shuffle. X is not carrying `B_{t+1}` into the feature row.

---

## Mentor note — 95% is still blocked

Sealed test logistic 180 is **64.87%**. Persistence on the same rows is **58.56%**. The high-confidence fifth of the test is **78.9%** — the best honest slice — and that is still **16 pp** short of 95%, on 20% of bars, with bands locked on val.

Phase A: 41.36% of consecutive 5m typical-price signs flip in the unseen next bar. Placebo 49.0% says this X does not see that bar. Adding blocks 8–11 did not change test accuracy vs 128. **Do not plan on 95%.** This is a ~15 pp-over-coin, well-calibrated 5m direction probability. It is not a 95% call.

---

## What this turn did not do

- No hyperparameter search.
- No threshold picked on test.
- Did not train LightGBM/MLP on test.
- Did not change `y_5m_dir`.
- Did not start a trading layer.
