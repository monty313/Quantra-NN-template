# Round 4 — two-bar labels A and B

No new columns. Same 180-col machine. fold_60_20, seed 0, logistic C=1, train-only scaler.
Sealed newest 20% unscored. No LightGBM/MLP. PR #1 untouched. Observation only.

JSON: `/cursor/stores/bc-63d4488e-eaab-49e7-b822-99139192b098/docs/round4-two-bar.json`

## Base to think from

**Before the numbers.** Two-bar y is the honest next question after Round 3: if 1-bar
next-close-vs-typical is just this close vs this typical (rule 61.90%, model −0.15 pp),
then looking at bar t+2 either (1) the known offset still leaks into y because FX 5m
closes stay near this close, so A is a weaker copy of R (~55–58%), or (2) two bars is
enough that R dies and we finally measure leftover skill. CEO G1 said leftover 5m
direction on this X is a coin; A-residual vs 50% is that test on a longer horizon.
B is two-bar close-to-close. One-bar close-to-close was already ~50.9%. Belief: B is
still a coin, R is the wrong baseline for B, and close-to-close persistence
(sign(close_t − close_{t−1})) is the fair sibling — also near 50%. The 180 will still
agree with R on A because X is a close-vs-mean machine. No new columns can change that.


**What the numbers did to that belief.**

A: 180 is 58.27% vs rule 58.41% (−0.14 pp). Agree-with-rule 91.66%; disagree acc 49.14%. Residual (y≠R) n=27,228 acc **9.85%** vs 50% (−40.15 pp). R still owns A at two bars — the known offset survives a 10-minute wait, just weaker (58.4% vs Round 3 61.9%). Belief (1) held. Belief (2) did not: two bars did not kill R. A-residual is far below 50% because the 180 still copies R, so it is systematically wrong on R’s misses. A coin would be 50%. CEO G1 stands: no leftover 5m direction skill. B: 180 is 51.12% vs rule 49.66% (+1.46 pp), agree-with-rule 30.34%. One-bar close-to-close was ~50.9%. B is still a coin. Two-bar close-to-close did not create a forecast. Belief on B held. R is the wrong baseline for B (agrees 30%). B persist sign(close_t − close_t−1) = 49.69%. Placebos A 49.74% / 50.15%; B 50.09% / 49.85%. Do not add columns.

## Machine

- Prediction time: right after a 5m bar closes, once every 5 minutes.
- Purge: **10 minutes**. Embargo: **10 minutes**. Train cutoff = val_start − 20 m. Both at the 10-minute (2-bar) floor. Stage 1 1h-head embargo of 60m is not used here.
- n parquet rows: 328,983. Sealed: 65,797 (not scored).
- Missing t+2 on all parquet rows: 360.
- Leak tests (pytest): shift y one 5m bar → X unchanged; mutate close of bar t+2 → y flips, X at t unchanged.
- Labels never in X. `y_5m_dir` and Round 3 y kept.

## Side by side vs Round 3 (one-bar)

| | 1-bar (Round 3 / Round 2) | 2-bar (this) logistic 180 | 2-bar rule | 180 − rule |
|---|---|---|---|---|
| A close vs typical | 61.75% / rule 61.90% | 58.27% | 58.41% | -0.14 pp |
| B close vs close | ~50.9% close-to-close 1-bar | 51.12% | 49.66% | +1.46 pp |

## A — `y_5m_close2_vs_typ`

- train n (purge + finite y): **196,421**
- val n finite y: **65,471**
- ties dropped train / val: **733** / **269**
- missing t+2 dropped train / val: **232** / **57**

| Model / baseline | n | Acc |
|---|---|---|
| **Logistic 180** | 65,471 | **58.27%** |
| Logistic 128 | 65,471 | 58.24% |
| Rule: this close > this typical else 0 | 65,471 | **58.41%** |
| Majority (train mode = 1) | 65,471 | 50.80% |
| prev5m_dir (zeros dropped) | 64,993 | 55.09% |

**KEY: logistic 180 minus rule = -0.14 pp.**
Logloss 180 0.6742. Logloss 128 0.6744.

### Agree / disagree with the rule

- Agree n=60,011 acc=59.10% (rate 91.66%)
- Disagree n=5,460 acc=49.14%

### Residual of A (y ≠ rule) vs 50%

- n=27,228 model acc=**9.85%** (vs 50%: −40.15 pp). The 180 still copies R, so it is wrong on R’s misses. A coin would be 50%. Not leftover skill.

### 10-bin calibration (equal-count, P(y=1), logistic 180)

| Bin | n | Mean pred | Actual rate |
|---|---|---|---|
| 1 | 6,547 | 0.3469 | 0.3440 |
| 2 | 6,547 | 0.4043 | 0.4008 |
| 3 | 6,547 | 0.4359 | 0.4297 |
| 4 | 6,547 | 0.4633 | 0.4692 |
| 5 | 6,547 | 0.4894 | 0.4821 |
| 6 | 6,547 | 0.5150 | 0.5218 |
| 7 | 6,547 | 0.5408 | 0.5667 |
| 8 | 6,547 | 0.5684 | 0.5815 |
| 9 | 6,547 | 0.6006 | 0.6155 |
| 10 | 6,548 | 0.6604 | 0.6692 |

### Acc by hour

| Hour | n | Acc |
|---|---|---|
| 00 | 1,975 | 56.51% |
| 01 | 2,339 | 59.00% |
| 02 | 2,486 | 57.52% |
| 03 | 2,592 | 56.13% |
| 04 | 2,572 | 57.08% |
| 05 | 2,559 | 59.28% |
| 06 | 2,543 | 59.81% |
| 07 | 2,507 | 57.40% |
| 08 | 2,567 | 58.47% |
| 09 | 2,599 | 56.79% |
| 10 | 2,614 | 58.53% |
| 11 | 2,592 | 58.45% |
| 12 | 2,580 | 57.56% |
| 13 | 2,572 | 57.70% |
| 14 | 2,575 | 57.71% |
| 15 | 2,566 | 57.79% |
| 16 | 2,843 | 57.93% |
| 17 | 3,279 | 59.56% |
| 18 | 3,267 | 58.98% |
| 19 | 3,226 | 58.80% |
| 20 | 3,245 | 60.09% |
| 21 | 3,256 | 59.92% |
| 22 | 3,223 | 58.11% |
| 23 | 2,894 | 57.53% |

### Acc by weekday

| Weekday | n | Acc |
|---|---|---|
| Mon | 4,826 | 59.03% |
| Tue | 15,081 | 58.56% |
| Wed | 14,943 | 58.05% |
| Thu | 15,279 | 58.03% |
| Fri | 15,342 | 58.20% |

### Placebo (frozen 180 preds vs shuffled val y)

- Shuffled-label: n=65,471 acc=49.74%
- Shuffled-time 1h blocks: n=65,471 acc=50.15%

## B — `y_5m_close2_vs_close`

- train n (purge + finite y): **193,689**
- val n finite y: **64,311**
- ties dropped train / val: **3,465** / **1,429**
- missing t+2 dropped train / val: **232** / **57**

| Model / baseline | n | Acc |
|---|---|---|
| **Logistic 180** | 64,311 | **51.12%** |
| Logistic 128 | 64,311 | 51.01% |
| Rule: this close > this typical else 0 | 64,311 | **49.66%** |
| Majority (train mode = 1) | 64,311 | 50.70% |
| prev5m_dir (zeros dropped) | 63,846 | 49.82% |
| persist sign(close_t − close_t−1) | 62,283 | 49.69% |

**KEY: logistic 180 minus rule = +1.46 pp.**
Logloss 180 0.6934. Logloss 128 0.6933.

### Agree / disagree with the rule

- Agree n=19,515 acc=51.27% (rate 30.34%)
- Disagree n=44,796 acc=51.05%

### 10-bin calibration (equal-count, P(y=1), logistic 180)

| Bin | n | Mean pred | Actual rate |
|---|---|---|---|
| 1 | 6,431 | 0.4507 | 0.4838 |
| 2 | 6,431 | 0.4692 | 0.4981 |
| 3 | 6,431 | 0.4794 | 0.4883 |
| 4 | 6,431 | 0.4881 | 0.5103 |
| 5 | 6,431 | 0.4958 | 0.4985 |
| 6 | 6,431 | 0.5036 | 0.5180 |
| 7 | 6,431 | 0.5113 | 0.5153 |
| 8 | 6,431 | 0.5201 | 0.5161 |
| 9 | 6,431 | 0.5312 | 0.5124 |
| 10 | 6,432 | 0.5526 | 0.5295 |

### Acc by hour

| Hour | n | Acc |
|---|---|---|
| 00 | 1,888 | 58.05% |
| 01 | 2,249 | 50.42% |
| 02 | 2,413 | 51.06% |
| 03 | 2,550 | 51.65% |
| 04 | 2,517 | 50.26% |
| 05 | 2,502 | 51.72% |
| 06 | 2,471 | 50.95% |
| 07 | 2,419 | 52.00% |
| 08 | 2,517 | 49.70% |
| 09 | 2,572 | 52.64% |
| 10 | 2,592 | 50.31% |
| 11 | 2,562 | 50.59% |
| 12 | 2,546 | 50.75% |
| 13 | 2,548 | 50.63% |
| 14 | 2,556 | 51.33% |
| 15 | 2,545 | 52.46% |
| 16 | 2,814 | 53.09% |
| 17 | 3,259 | 49.13% |
| 18 | 3,223 | 49.64% |
| 19 | 3,195 | 51.33% |
| 20 | 3,197 | 51.02% |
| 21 | 3,207 | 49.27% |
| 22 | 3,150 | 50.54% |
| 23 | 2,819 | 51.33% |

### Acc by weekday

| Weekday | n | Acc |
|---|---|---|
| Mon | 4,746 | 51.47% |
| Tue | 14,815 | 51.33% |
| Wed | 14,694 | 51.02% |
| Thu | 15,002 | 50.89% |
| Fri | 15,054 | 51.12% |

### Placebo (frozen 180 preds vs shuffled val y)

- Shuffled-label: n=64,311 acc=50.09%
- Shuffled-time 1h blocks: n=64,311 acc=49.85%

## Notes

- 0/1 mapped to −1/+1 only at `LogisticPolicy.fit`.
- Rule baseline is always this bar’s close vs this bar’s typical (known at t), for both labels.
- B also reports sign(close_t − close_{t−1}) persistence; zeros/missing dropped from that n.
- Residual of A is model acc on val rows where y_A ≠ rule, vs 50%. Not a new feature.
