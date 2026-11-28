# Phase B — feature append (stop here)

**Phase B only. No Phase C/D. No training. Test partition was not touched for tuning.**

Branch: `cursor/stage1-market-state-a365` on `monty313/Quantra-NN-template`.
Specs: HolyGrail Feature Specification v1.0 + `rsi_sma_bb_feature_specs_v1.0.pdf`.
Timing: **v1.0 developing-bar** (live_1m + developing 5m/30m/1h), same as columns 1–128.

`y_5m_dir` is unchanged: alias of `y_5m` = `sign(TP(B_{t+1}) − TP(B_t))`. Exact 0 dropped. Not recomputed.

Replay: `python -m pytest tests/test_t1_t44.py tests/test_structure_labels.py tests/test_phase_b_features.py -q` → **59 passed**. Did **not** run `build_dataset.py` or `train_eval.py`. Existing `artifacts/scaler.joblib` / `train_results.json` / processed parquet are still the 128-col Phase A run.

---

## Totals

| Item | Value |
|---|---|
| X columns | **180** (128 HolyGrail v1.0 + 52 Phase B) |
| First 128 names and order | **unchanged** |
| RobustScaler gaps | **67** (46 + 9 rsisma + 3 rbb_mid + 9 s4/s20) |
| `prev5m_dir` | raw ±1/0, **not** scaled |
| FEATURE_VERSION | `1.1` |
| `y_5m_dir` | unchanged |

---

## Column map (1-based)

### Block 8 — `rsi_sma_trio` — cols 129–147 (19)

Per TF, `p ∈ {3, 7, 14}`: `rsisma{p}_bin` then `rsisma{p}_gap`. Timeframes **5m, then 30m, then 1h**, then `rsisma_label`.

RSI on **Close** (Wilder / iRSI). SMA 5 of that RSI. Separate from block-3 RSI-on-typical.

| Name | Formula | Scale |
|---|---|---|
| `rsisma{p}_bin_{tf}` | `sign(RSI{p} − SMA5(RSI{p}))` | raw |
| `rsisma{p}_gap_{tf}` | `RSI{p} − SMA5(RSI{p})` | RobustScaler (9) |
| `rsisma_label` | +1 if 5m has ≥2 of 3 bins +1 **and** at least one HTF also has ≥2 of 3 +1; −1 mirror; else 0 | raw |

### Block 9 — `rsi_bb_reversion` — cols 148–160 (13)

RSI 4 on Close. Bollinger **period 20, deviation 1** on the RSI 4 line. Per TF: pctb, mid_gap, mid_bin, band_state. Then `rbb_label`.

| Name | Formula | Scale |
|---|---|---|
| `rbb_pctb_{tf}` | `(RSI4 − Lower) / (Upper − Lower)` | raw |
| `rbb_mid_gap_{tf}` | `RSI4 − Mid` | RobustScaler (3) |
| `rbb_mid_bin_{tf}` | `sign(RSI4 − Mid)` | raw |
| `rbb_band_state_{tf}` | +1 if RSI4 > Upper, −1 if RSI4 < Lower, else 0 | raw |
| `rbb_label` | +1 if both HTF mid_bin = +1 **and** 5m band_state = −1; −1 mirror; else 0 | raw |

### Block 10 — `sma4hl_rsi_bb` — cols 161–179 (19)

SMA4 High, SMA4 Low, SMA20 Close, ATR14 same TF, RSI4+BB same as block 9. Price = Close.

| Name | Formula | Scale |
|---|---|---|
| `s4_hi_gap_{tf}` | `(Close − SMA4H) / ATR14` | RobustScaler |
| `s4_lo_gap_{tf}` | `(Close − SMA4L) / ATR14` | RobustScaler |
| `s4_state_{tf}` | +1 Close above both SMA4H and SMA4L, −1 below both, 0 between/equal | raw |
| `s20_gap_{tf}` | `(Close − SMA20) / ATR14` | RobustScaler |
| `s4rbb_pctb_{tf}` | same as `rbb_pctb_{tf}` | raw |
| `s4rbb_band_state_{tf}` | same as `rbb_band_state_{tf}` | raw |
| `s4hl_label` | +1 if both HTF `s4_state` = +1 **and** 5m `s20_gap > 0` **and** 5m band_state = −1; −1 mirror; else 0 | raw |

Nine scaler gaps: hi/lo/s20 × 3 TF.

### Block 11 — col 180 — `prev5m_dir`

```
prev5m_dir = sign(TP5(k) − TP5(k−1))
k = last COMPLETED 5m bar at decision_time
TP = (H + L + C) / 3
```

- Raw, completed bars only. Developing TP of the open 5m is never read.
- At the last 1m of `B_t`, `decision_time = B_t` close, so `k = B_t` and `prev5m_dir = sign(TP(B_t) − TP(B_{t−1}))`.
- Earlier inside `B_t`, `k = B_{t−1}` (last bar whose close ≤ decision).
- Equality → **0** (allowed in this feature). Missing k or k−1 → NaN, row invalid.
- **Not** `y_5m_dir`. That is `sign(TP(B_{t+1}) − TP(B_t))`.

---

## BB std ddof (report)

**ddof = 0** (population, divide by n). Same as MT5 `iBands` / `iStdDev`.

We already used 0 on the price Bollinger (HolyGrail T36, `BB_DDOF = 0`). Phase B RSI Bollinger uses the same `BB_DDOF`. Sample std (ddof=1) does not match iBands.

---

## Timing conflict (documented, not mixed)

`rsi_sma_bb_feature_specs_v1.0` says closed Bar Index 1 only, never index 0, and HTF from the last **fully closed** 30m/1h.

This turn ordered **v1.0 developing-bar timing**. Implementation uses the same asof developing last bar as columns 1–128: previous F-bars frozen, current F-bar rebuilt from completed 1m through decision. One train/live path. Not MT5 `CopyBuffer` shift 1 closed-only. Not a mix of developing LTF + closed HTF.

---

## Tests

| Check | Result |
|---|---|
| T1–T44 + structure labels (updated 180 / 67) | PASS |
| First 128 names unchanged; cols 129–180 as mapped | PASS |
| Scaler 67; `prev5m_dir` / bins / pctb / labels not scaled | PASS |
| T24: `y_5m_dir` not in X | PASS |
| T3 + Phase B: mutate 1m after t does not change the row, including new cols | PASS |
| `prev5m_dir` ignores remainder of the open 5m | PASS |
| Mutate next 5m bar does not change X at last 1m of this bar | PASS |
| Last 1m: `prev5m_dir = sign(TP this − TP prev)` from completed bars | PASS |
| `rsisma_label` / `rbb_label` / `s4hl_label` formulas | PASS |
| RSI-on-Close SMA gap matches closed iRSI at bar end | PASS |
| RSI BB ddof=0 | PASS |
| Equality → 0 | PASS |

---

## Mentor note — why ~95% 5m FX direction is still blocked

Phase B added state of the **already-seen** bars. It did not add a view of `B_{t+1}`.

| Fact (Phase A, EURUSD last-1m-of-5m test) | Number |
|---|---|
| Flip rate next 5m TP sign ≠ this 5m TP sign | **41.36%** |
| Persistence ceiling (always continue) | **58.64%** |
| Logistic on 128-col `y_5m_dir` | **64.87%** |
| To hit 95% if every continuation stays right | need **~87.9%** of *flips* right |

`prev5m_dir` **is** the persistence bit: this completed 5m vs the previous completed 5m. That is A1’s 58.64% feature, not the label. The label is the **next** closed 5m, which alignment tests still cannot see (mutating `B_{t+1}` leaves X, including blocks 8–11, unchanged).

Blocks 8–10 are RSI / SMA / BB of developing Close through decision. Same information family as the first 128: functions of bars that have already started or finished. They can help a few points if they describe *when* continuation is more likely. They cannot remove the 41% of rows whose sign lives in an unseen bar.

Placebo (predict `B_{t+2}` vs `B_{t+1}` from X at end of `B_t`) was **51.36%**. Shuffle was **49.22%**. Those still apply: the new columns are leak-checked the same way.

**Do not plan on 95%.** A honest later target is a few points above 64.9% on this same label after a train-only rebuild (Phase C), or a different question.

---

## What this turn did not do

- No Phase C (rebuild parquet / refit scaler / train on train+val).
- No Phase D.
- No `train_eval.py`. No look at test for tuning.
- `y_5m_dir` definition not touched.
- Processed 128-col artifacts left in place on purpose.
