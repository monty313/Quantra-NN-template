# Phase A — 5m direction audit

**Stop here. No new features. Phase B–D not started.**

EURUSD, last 1m of each 5m bar. **n_test = 117,598** (same row set as the 5m logistic run). n_train = 217,642. Developing-bar v1.0 timing unchanged. Python.

Replay: `python scripts/phase_a_5m_audit.py` → `artifacts/phase_a_5m_audit.json`.

---

## Exact `y_5m` / `y_5m_dir` (from `holygrail/labels.py`, not memory)

`y_5m_dir` is an **alias of `y_5m`**. Same array. Not recomputed. Not a new target.

From `structure_direction(m1, "5m")`:

1. Completed 5m bars = `resample_ohlc(m1, "5min")`: left-closed, labeled by **bar open**, empty bins dropped.
2. Typical price of a completed 5m bar: `tp = (high + low + close) / 3`.
3. For each 1m row: `current_open = 1m_open.floor("5min")`, `next_open = current_open + 5 minutes`.
4. **`y = sign(tp[next_open] − tp[current_open])` ∈ {−1, +1}.**
5. Missing this bar, missing next bar, or **exact-zero** move → NaN, **drop**, never class 0.

No 0.5× ATR. Std/ATR/RSI stay in X only.

At the last 1m of a 5m bar: 1m open is `B_t` open + 4m; **decision_time = 1m open + 1m = `B_t` close = `B_{t+1}` open**. Label window is the next 5m bar `[B_{t+1} open, B_{t+1} close)`. X at that snapshot may use 1m candles with open ≤ this 1m open (complete by decision_time). It may not use the 1m that **opens** at decision_time (first candle of `B_{t+1}`).

---

## Why ~95% 5m FX direction is almost certainly impossible

| Fact | Number |
|---|---|
| Flip rate (next 5m TP sign ≠ this 5m TP sign) | **41.36%** (48,263 / 116,679 consecutive) |
| Persistence ceiling (always “continue”) | **58.64%** |
| Logistic on real `y_5m_dir` | **64.87%** (+6.23 pp over persist) |
| Gap from logistic to 95% | **30.13 pp** |
| To hit 95% if every continuation stays right | need **87.9%** of *flips* right: (0.95−0.5864) / 0.4136 |
| Those flips are `B_{t+1}` vs `B_t`. `B_{t+1}` is the label window. Alignment: mutating it **does not change X** |
| Placebo (predict `B_{t+2}` vs `B_{t+1}` from X at end of `B_t`) | **51.36%** |
| Close-to-close next 5m (this close vs next close) logistic | **50.99%** |

95% would mean X already almost knows the next 5m typical. Placebo, shuffle, and the future-mutate test say it does not. The 41% flip rate is the floor you cannot remove without seeing the next bar. **Do not plan on 95%.** A honest later target is a few points above 64.9% on this same label, or a different question (not 5m FX sign).

---

## Trial log (Phase A = trials A1–A7)

| ID | Trial | n | Acc / rate | vs 50% | Stop? |
|---|---|---|---|---|---|
| A1 | Persistence: next dir = this 5m dir | 116,679 consec. (919 gaps skipped) | **58.64%** | +8.64 pp | no |
| A2 | Lag-1 Pearson of `y_5m_dir` | 116,679 | **ρ = 0.173** (= 2×continue − 1) | — | no |
| A2b | Flip / continue | 48,263 / 68,416 | **41.36% / 58.64%** | — | no |
| A3 | Logistic **close-to-close** 5m (ties dropped; **not** `y_5m_dir`) | test 114,795 (2,803 ties/missing) | **50.99%** | +0.99 pp | no |
| A3b | Persistence of close-to-close | 111,249 | **49.44%** | −0.56 pp | no |
| A4 | Placebo: `y[t] ← y_5m_dir[t+1]` (extra 5m bar), refit logistic | test 116,679 | **51.36%** | +1.36 pp | **no** (not > 52%) |
| A5 | Shuffle: train labels block-permuted (12×5m = 1h), real test y | 117,598 | **49.22%** | −0.78 pp | **no** (not > 52%) |
| A6 | Alignment (T3/T13 + EURUSD window) | — | X max abs diff **0** after next-bar mutate; y flipped +1→−1 | — | no |
| A7 | Go / no-go | — | placebo and shuffle both ≤ 52% | — | **GO for later phases; STOP this turn** |
| — | Real logistic `y_5m_dir` (sanity) | 117,598 | **64.87%** | +14.87 pp | matches prior run |

Close-to-close vs TP: on the 114,795 close-to-close rows, sign(close next−close this) **equals** `y_5m_dir` on **82.9%** of rows. They are not the same label. Close-to-close is a coin; TP-to-TP is not.

---

## A1–A2 persistence / lag-1

On consecutive 5m bars (gap ≠ 5 minutes → skip, not fill):

- Continue: 68,416 (**58.64%**)
- Flip: 48,263 (**41.36%**)
- Persistence acc = continue rate = **58.64%**
- Lag-1 Pearson **0.1727** (matches 2×0.5864 − 1 = 0.1727)

Weak positive autocorrelation. Not a random walk. Not a strong trend machine.

---

## A3 close-to-close (comparison only)

`sign(close[next 5m] − close[this 5m])`. Exact ties dropped. **Did not replace `y_5m_dir`.**

| | Acc |
|---|---|
| Logistic | **50.99%** (n_test=114,795) |
| Persistence | **49.44%** |

This close does not forecast the next close. The 64.9% on `y_5m_dir` is **typical-price continuation**, not “we can see the next close.”

---

## A4 placebo

`y_placebo[t] = y_5m_dir[t+1]` when the next 5m bar is exactly 5 minutes later = sign(TP(`B_{t+2}`) − TP(`B_{t+1}`)). Refit logistic on that y. Test acc **51.36%**.

Rule: stop if **well above 52%**. 51.36% is not. Slightly over 50% with a +1-biased predictor (pred up 57.0%). Not a leak of `B_{t+2}`.

Caveat (honest): a placebo of `B_{t+2}` vs `B_{t+1}` would still be ~50% even if X leaked **`B_{t+1}`**, because `B_{t+2}` is still unseen. The **mutate-next-bar** test in A6 is the leak check for the actual label window.

---

## A5 shuffle

Train `y_5m_dir` permuted in blocks of **12** consecutive 5m rows (1 hour). X order unchanged. Test y real. Acc **49.22%**.

Rule: stop if well above 52%. **49.22% does not stop.** Mapping X→y was destroyed. The 64.9% on real y is not an X-only artifact.

---

## A6 alignment

**Clocks**

| Clock | What |
|---|---|
| 1m CSV timestamp | bar **open** (naive broker-server) |
| decision_time | open + 1m = 1m **close** |
| current 5m `B_t` | `floor(1m open, 5min)` |
| last 1m of `B_t` | 1m open = `B_t` + 4m; decision = `B_t` close = next 5m **open** |
| `y_5m_dir` | sign(TP(`B_{t+1}`) − TP(`B_t`)); TP(`B_{t+1}`) uses 1m opens in `[B_{t+1}, B_{t+1}+5m)` which are **after** this decision |
| X developing 5m | 1m candles with open ≤ this 1m open only (`holygrail/timing.py`, `asof.py`) |

**Tests**

- T3/T4: mutate 1m **after** t → row t X unchanged. **Passed.**
- T13: prefix build last row = full build at that index. **Passed.**
- Structure leak test: mutate after next 5m close → `y_5m` at t unchanged. **Passed.**
- EURUSD window (2024-10-14 12:04 last 1m of 12:00 bar): crush next 5m 1m bars (12:05–12:09). **X max abs diff = 0.0. `y_5m_dir` +1 → −1.** Mutating earlier 1m *inside* this 5m bar **does** move X (developing), as required.

Features at t do not contain the label window. They **do** contain this bar’s completed TP at the last 1m (by design). That is why persistence and CCI-continuation show up. It is not a future leak.

---

## A7 decision

Placebo 51.36% and shuffle 49.22% are **not** well above 52%. **Do not stop the program for leak.** **Stop this turn anyway** (Phase A only). No features added.

What actually blocks ~95%: the **41% 5m TP reversals** live in `B_{t+1}`, which X is forbidden to see. Logistic already took the linear continuation juice (+6 pp over persist). Close-to-close being 51% says even a “next close” question is empty. 95% is the wrong number for this label.

---

## What was not done

- No new X columns.
- Phase B (RSI/SMA/BB spec) not started.
- `y_5m_dir` not rewritten. Alias only.
