# Round 5 — feature ranking L1–L4

No new columns. fold_60_20, seed 0, logistic 180 C=1. Sealed unscored. PR #1 untouched.
Purge = embargo = **5m** on L1/L2, **10m** on L3/L4. Permutation: 10 shuffles/column. Drop-one-group refit.
No-signal = |drop| ≤ 2×SD. Empty families are not in the 180.

## Base to think from

**Before.** L1/L3 (vs typical) should be owned by SMA/gap and Bollinger — the close-vs-mean offset. CCI, RSI, StdDev, HTF, prev5m_dir should be no-signal. L2/L4 (vs close) should show no family with a real drop: one-bar close-to-close was ~50.9%, two-bar 51.1%. Anything that still moves L2/L4 is a genuine direction signal. Empty families in the 180: HTF 15m, bar-shape, volume (not in this X).

**After.** L1 180 61.77% vs rule 61.90% (-0.13 pp). L2 180 50.92% vs rule 49.80% (+1.12 pp). L3 180 58.27% vs rule 58.41% (-0.14 pp). L4 180 51.12% vs rule 49.66% (+1.46 pp). L1/L3 are the offset: drop SMA/gap and val falls 1.35 / 1.16 pp; Bollinger perm is large but drop-Bollinger is ~0 because SMA/gap already copies it. L2/L4 stay coins. Drop-one-group max |Δ| is -0.20 / -0.20 pp — no family is load-bearing on close-to-close. 2-SD perm flags many tiny L2/L4 drops; that is shuffle noise, not direction skill. Placebo family |drop| avg 0.46 pp.

## Model vs rule

| Label | Question | n | 180 | Rule | 180−rule | Majority | Placebo acc |
|---|---|---|---|---|---|---|---|
| L1 | next close vs typical | 65,394 | **61.77%** | 61.90% | -0.13 pp | 50.78% | 48.65% |
| L2 | next close vs close | 63,665 | **50.92%** | 49.80% | +1.12 pp | 50.31% | 50.10% |
| L3 | t+2 close vs typical | 65,471 | **58.27%** | 58.41% | -0.14 pp | 50.80% | 49.69% |
| L4 | t+2 close vs close | 64,311 | **51.12%** | 49.66% | +1.46 pp | 50.70% | 50.10% |

## Families across L1–L4 (permutation drop pp)

| Family | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| SMA/gap | **+10.02** | **+0.75** | **+7.20** | +0.37* |
| Bollinger | **+12.23** | **+0.59** | **+8.40** | **+0.85** |
| RSI | **+1.99** | **+0.93** | **+2.13** | **+1.36** |
| CCI | **+0.15** | **+0.40** | **+0.20** | **+0.41** |
| StdDev | -0.01* | +0.02* | -0.01* | +0.07* |
| HTF 15m | empty | empty | empty | empty |
| HTF 30m | **+4.27** | **+0.96** | **+4.43** | **+1.22** |
| HTF 1h | **+3.29** | **+0.33** | **+3.47** | +0.09* |
| prev5m_dir | -0.01* | +0.01* | -0.02* | +0.05* |
| bar-shape | empty | empty | empty | empty |
| volume | empty | empty | empty | empty |

* = no signal (|drop| ≤ 2×SD). Bold = 2-SD signal. HTF 30m/1h overlap SMA/BB/RSI (not extra columns).

## Real direction vs offset-only

2-SD flags **71** columns on L2/L4 because shuffle SD is tiny (~0.1 pp). Drop-one-group on L2/L4 is ≤ **0.20 pp** for every family. Treat ≥ **1.0 pp** perm drop as the practical floor (placebo column |drop| is 0.1–0.4 pp).

L2/L4 perm drop ≥ 1.0 pp:

| Column | Family | L2 | L4 | L1 (offset) | L3 (offset) |
|---|---|---|---|---|---|
| `s20_gap_5m` | SMA/gap | +1.06 | +0.32 | +10.49 | +7.14 |
| `bbs_mid_gap_30m` | Bollinger | +0.37 | +1.06 | +2.92 | +2.94 |

Both are close-vs-mean gaps that also dominate L1. Not a new direction skill.

Offset-only (signal on L1/L3, none on L2/L4): **25** columns. Top 10:

| Column | Family | L1 drop | L3 drop |
|---|---|---|---|
| `ltf_price_sma20_gap_5m` | SMA/gap | +9.43 | +6.96 |
| `lad_g1_30m` | SMA/gap | +0.33 | +1.51 |
| `lad_g2_30m` | SMA/gap | +0.31 | +1.12 |
| `lad_g1_1h` | SMA/gap | +1.07 | +0.09 |
| `bbs_pctb_5m` | Bollinger | +0.15 | +0.64 |
| `rsisma7_gap_1h` | RSI | +0.00 | +0.68 |
| `bbs_mid_gap_1h` | Bollinger | +0.06 | +0.40 |
| `rbb_mid_gap_5m` | Bollinger | +0.06 | +0.34 |
| `lad_g3_5m` | SMA/gap | +0.04 | +0.34 |
| `lad_50gap_1h` | SMA/gap | +0.12 | +0.21 |

## L1 next close vs typical

n=65,394 ties=403 purge=embargo=5m. 180 61.77% vs rule 61.90%.

Top 20 permutation

| Rank | Column | Family | Drop pp | SD | Flag |
|---|---|---|---|---|---|
| 1 | `bbs_sma20_gap_5m` | Bollinger | +10.64 | 0.14 | signal |
| 2 | `s20_gap_5m` | SMA/gap | +10.49 | 0.12 | signal |
| 3 | `ltf_price_sma20_gap_5m` | SMA/gap | +9.43 | 0.15 | signal |
| 4 | `bbs_mid_gap_5m` | Bollinger | +6.61 | 0.16 | signal |
| 5 | `bbs_mid_gap_30m` | Bollinger | +2.92 | 0.13 | signal |
| 6 | `sma_bc_gap_5m` | SMA/gap | +2.18 | 0.14 | signal |
| 7 | `lad_50gap_5m` | SMA/gap | +1.90 | 0.08 | signal |
| 8 | `lad_g2_5m` | SMA/gap | +1.79 | 0.10 | signal |
| 9 | `lad_g1_5m` | SMA/gap | +1.68 | 0.08 | signal |
| 10 | `sma_ab_gap_5m` | SMA/gap | +1.38 | 0.11 | signal |
| 11 | `lad_g1_1h` | SMA/gap | +1.07 | 0.09 | signal |
| 12 | `s4_lo_gap_5m` | SMA/gap | +0.85 | 0.09 | signal |
| 13 | `sma_bc_gap_1h` | SMA/gap | +0.81 | 0.08 | signal |
| 14 | `sma_cd_gap_5m` | SMA/gap | +0.60 | 0.06 | signal |
| 15 | `lad_50gap_30m` | SMA/gap | +0.54 | 0.10 | signal |
| 16 | `s4_hi_gap_30m` | SMA/gap | +0.49 | 0.07 | signal |
| 17 | `rsisma14_gap_5m` | RSI | +0.43 | 0.10 | signal |
| 18 | `rsi14_dist50_5m` | RSI | +0.39 | 0.09 | signal |
| 19 | `sma_bc_gap_30m` | SMA/gap | +0.36 | 0.10 | signal |
| 20 | `rsisma7_gap_30m` | RSI | +0.35 | 0.08 | signal |

Bottom 20 permutation (smallest drop)

| Rank | Column | Family | Drop pp | SD | Flag |
|---|---|---|---|---|---|
| 161 | `s714_gap_5m` | SMA/gap | -0.04 | 0.03 | no signal |
| 162 | `lad_bonus` | SMA/gap | -0.04 | 0.01 | signal |
| 163 | `htf2_cci10_bin` | CCI | -0.04 | 0.02 | signal |
| 164 | `htf2_cci100_gap` | CCI | -0.02 | 0.03 | no signal |
| 165 | `sma_cd_gap_1h` | SMA/gap | -0.02 | 0.03 | no signal |
| 166 | `htf2_std14_bin` | StdDev | -0.02 | 0.03 | no signal |
| 167 | `lad_p3_1h` | SMA/gap | -0.02 | 0.03 | no signal |
| 168 | `htf1_cci30_bin` | CCI | -0.02 | 0.01 | no signal |
| 169 | `bbs_sma20_gap_30m` | Bollinger | -0.02 | 0.02 | no signal |
| 170 | `htf1_std7_gap` | StdDev | -0.02 | 0.01 | no signal |
| 171 | `htf1_cci10_gap` | CCI | -0.02 | 0.02 | no signal |
| 172 | `rsisma7_bin_1h` | RSI | -0.02 | 0.01 | no signal |
| 173 | `s4rbb_pctb_1h` | Bollinger | -0.01 | 0.01 | no signal |
| 174 | `bbs_sma20_gap_1h` | Bollinger | -0.01 | 0.04 | no signal |
| 175 | `s714_gap_1h` | SMA/gap | -0.01 | 0.03 | no signal |
| 176 | `htf2_std7_bin` | StdDev | -0.01 | 0.01 | no signal |
| 177 | `sma_ab_bin_1h` | SMA/gap | -0.01 | 0.02 | no signal |
| 178 | `lad_p1_30m` | SMA/gap | -0.01 | 0.01 | no signal |
| 179 | `ltf_cci10_gap` | CCI | -0.01 | 0.02 | no signal |
| 180 | `htf2_cci30_bin` | CCI | -0.01 | 0.03 | no signal |

Families (perm)

| Family | n | Drop pp | SD | |
|---|---|---|---|---|
| Bollinger | 32 | +12.23 | 0.21 | signal |
| SMA/gap | 68 | +10.02 | 0.18 | signal |
| HTF 30m | 55 | +4.27 | 0.15 | signal |
| HTF 1h | 55 | +3.29 | 0.09 | signal |
| RSI | 40 | +1.99 | 0.13 | signal |
| CCI | 27 | +0.15 | 0.04 | signal |
| StdDev | 12 | -0.01 | 0.03 | no signal |
| prev5m_dir | 1 | -0.01 | 0.02 | no signal |
| HTF 15m | 0 | — | — | empty |
| bar-shape | 0 | — | — | empty |
| volume | 0 | — | — | empty |

Drop-one-group retrain (Δ acc pp; negative = group was useful)

| Family | n removed | Val acc | Δ pp |
|---|---|---|---|
| SMA/gap | 68 | 60.42% | -1.35 |
| Bollinger | 32 | 61.78% | +0.01 |
| prev5m_dir | 1 | 61.79% | +0.02 |
| RSI | 40 | 61.80% | +0.02 |
| StdDev | 12 | 61.80% | +0.03 |
| CCI | 27 | 61.85% | +0.08 |

Top 10 |coef|

| Rank | Column | Family | Coef |
|---|---|---|---|
| 1 | `ltf_price_sma20_gap_5m` | SMA/gap | -3.781 |
| 2 | `bbs_sma20_gap_5m` | Bollinger | +1.617 |
| 3 | `s20_gap_5m` | SMA/gap | +1.617 |
| 4 | `bbs_mid_gap_5m` | Bollinger | +0.524 |
| 5 | `lad_g1_5m` | SMA/gap | -0.520 |
| 6 | `bbs_mid_gap_30m` | Bollinger | -0.437 |
| 7 | `rsi14_dist50_5m` | RSI | -0.398 |
| 8 | `sma_bc_gap_5m` | SMA/gap | +0.321 |
| 9 | `lad_50gap_5m` | SMA/gap | +0.281 |
| 10 | `lad_g2_5m` | SMA/gap | -0.273 |

Placebo (shuffled train y): val acc 48.65%; family mean |drop| 0.87 pp; column mean |drop| 0.40 pp.

## L2 next close vs close

n=63,665 ties=2,132 purge=embargo=5m. 180 50.92% vs rule 49.80%.

Top 20 permutation

| Rank | Column | Family | Drop pp | SD | Flag |
|---|---|---|---|---|---|
| 1 | `s20_gap_5m` | SMA/gap | +1.06 | 0.08 | signal |
| 2 | `bbs_sma20_gap_5m` | Bollinger | +0.99 | 0.19 | signal |
| 3 | `sma_ab_gap_5m` | SMA/gap | +0.87 | 0.20 | signal |
| 4 | `rsisma7_gap_5m` | RSI | +0.78 | 0.20 | signal |
| 5 | `lad_50gap_30m` | SMA/gap | +0.65 | 0.08 | signal |
| 6 | `sma_bc_gap_30m` | SMA/gap | +0.59 | 0.14 | signal |
| 7 | `lad_g1_5m` | SMA/gap | +0.58 | 0.16 | signal |
| 8 | `lad_g2_5m` | SMA/gap | +0.56 | 0.12 | signal |
| 9 | `rsi7_dist50_30m` | RSI | +0.48 | 0.12 | signal |
| 10 | `rsi7_dist50_5m` | RSI | +0.46 | 0.19 | signal |
| 11 | `rsi14_dist50_30m` | RSI | +0.44 | 0.11 | signal |
| 12 | `sma_bc_gap_1h` | SMA/gap | +0.44 | 0.11 | signal |
| 13 | `rsisma14_gap_1h` | RSI | +0.41 | 0.13 | signal |
| 14 | `bbs_mid_gap_30m` | Bollinger | +0.37 | 0.10 | signal |
| 15 | `rsisma7_gap_30m` | RSI | +0.36 | 0.08 | signal |
| 16 | `sma_bc_gap_5m` | SMA/gap | +0.33 | 0.11 | signal |
| 17 | `sma_cd_gap_30m` | SMA/gap | +0.32 | 0.08 | signal |
| 18 | `s4_lo_gap_30m` | SMA/gap | +0.31 | 0.12 | signal |
| 19 | `sma_ab_gap_30m` | SMA/gap | +0.29 | 0.13 | signal |
| 20 | `ltf_cci100_lvl` | CCI | +0.28 | 0.14 | no signal |

Bottom 20 permutation (smallest drop)

| Rank | Column | Family | Drop pp | SD | Flag |
|---|---|---|---|---|---|
| 161 | `htf2_cci100_bin` | CCI | -0.13 | 0.05 | signal |
| 162 | `bbs_mid_gap_1h` | Bollinger | -0.07 | 0.09 | no signal |
| 163 | `s4_hi_gap_1h` | SMA/gap | -0.05 | 0.05 | no signal |
| 164 | `s4_state_1h` | SMA/gap | -0.05 | 0.06 | no signal |
| 165 | `bbs_state_30m` | Bollinger | -0.05 | 0.07 | no signal |
| 166 | `rsi7_bin_1h` | RSI | -0.05 | 0.04 | no signal |
| 167 | `ltf_std14_gap` | StdDev | -0.05 | 0.04 | no signal |
| 168 | `rsisma14_bin_5m` | RSI | -0.04 | 0.05 | no signal |
| 169 | `s4rbb_pctb_1h` | Bollinger | -0.04 | 0.05 | no signal |
| 170 | `rbb_pctb_1h` | Bollinger | -0.04 | 0.04 | no signal |
| 171 | `bbs_pctb_1h` | Bollinger | -0.04 | 0.04 | no signal |
| 172 | `lad_p1_5m` | SMA/gap | -0.03 | 0.03 | no signal |
| 173 | `bbs_sma20_gap_30m` | Bollinger | -0.03 | 0.06 | no signal |
| 174 | `rbb_mid_bin_5m` | Bollinger | -0.03 | 0.03 | no signal |
| 175 | `s714_label` | SMA/gap | -0.03 | 0.03 | no signal |
| 176 | `lad_g1_1h` | SMA/gap | -0.03 | 0.11 | no signal |
| 177 | `rsisma_label` | RSI | -0.03 | 0.02 | no signal |
| 178 | `htf1_cci100_bin` | CCI | -0.02 | 0.02 | no signal |
| 179 | `lad_p2_30m` | SMA/gap | -0.02 | 0.02 | no signal |
| 180 | `bbs_label` | Bollinger | -0.02 | 0.02 | no signal |

Families (perm)

| Family | n | Drop pp | SD | |
|---|---|---|---|---|
| HTF 30m | 55 | +0.96 | 0.15 | signal |
| RSI | 40 | +0.93 | 0.23 | signal |
| SMA/gap | 68 | +0.75 | 0.12 | signal |
| Bollinger | 32 | +0.59 | 0.19 | signal |
| CCI | 27 | +0.40 | 0.10 | signal |
| HTF 1h | 55 | +0.33 | 0.12 | signal |
| StdDev | 12 | +0.02 | 0.05 | no signal |
| prev5m_dir | 1 | +0.01 | 0.01 | no signal |
| HTF 15m | 0 | — | — | empty |
| bar-shape | 0 | — | — | empty |
| volume | 0 | — | — | empty |

Drop-one-group retrain (Δ acc pp; negative = group was useful)

| Family | n removed | Val acc | Δ pp |
|---|---|---|---|
| RSI | 40 | 50.72% | -0.20 |
| Bollinger | 32 | 50.76% | -0.16 |
| SMA/gap | 68 | 50.81% | -0.11 |
| CCI | 27 | 50.83% | -0.09 |
| prev5m_dir | 1 | 50.94% | +0.01 |
| StdDev | 12 | 50.94% | +0.02 |

Top 10 |coef|

| Rank | Column | Family | Coef |
|---|---|---|---|
| 1 | `ltf_price_sma20_gap_5m` | SMA/gap | +0.415 |
| 2 | `rsisma7_gap_5m` | RSI | -0.199 |
| 3 | `s20_gap_5m` | SMA/gap | -0.197 |
| 4 | `bbs_sma20_gap_5m` | Bollinger | -0.197 |
| 5 | `rsi14_dist50_30m` | RSI | -0.187 |
| 6 | `bbs_mid_gap_5m` | Bollinger | +0.186 |
| 7 | `sma_bc_gap_30m` | SMA/gap | -0.167 |
| 8 | `sma_bc_gap_1h` | SMA/gap | +0.157 |
| 9 | `rsi7_dist50_30m` | RSI | -0.156 |
| 10 | `lad_g1_5m` | SMA/gap | -0.134 |

Placebo (shuffled train y): val acc 50.10%; family mean |drop| 0.16 pp; column mean |drop| 0.10 pp.

## L3 t+2 close vs typical

n=65,471 ties=326 purge=embargo=10m. 180 58.27% vs rule 58.41%.

Top 20 permutation

| Rank | Column | Family | Drop pp | SD | Flag |
|---|---|---|---|---|---|
| 1 | `s20_gap_5m` | SMA/gap | +7.14 | 0.08 | signal |
| 2 | `bbs_sma20_gap_5m` | Bollinger | +7.07 | 0.14 | signal |
| 3 | `ltf_price_sma20_gap_5m` | SMA/gap | +6.96 | 0.11 | signal |
| 4 | `bbs_mid_gap_5m` | Bollinger | +4.79 | 0.16 | signal |
| 5 | `bbs_mid_gap_30m` | Bollinger | +2.94 | 0.12 | signal |
| 6 | `sma_bc_gap_5m` | SMA/gap | +1.73 | 0.06 | signal |
| 7 | `lad_g1_30m` | SMA/gap | +1.51 | 0.11 | signal |
| 8 | `lad_g2_5m` | SMA/gap | +1.49 | 0.13 | signal |
| 9 | `lad_g1_5m` | SMA/gap | +1.44 | 0.16 | signal |
| 10 | `lad_50gap_5m` | SMA/gap | +1.33 | 0.12 | signal |
| 11 | `rsi7_dist50_30m` | RSI | +1.27 | 0.07 | signal |
| 12 | `lad_g2_30m` | SMA/gap | +1.12 | 0.10 | signal |
| 13 | `s4_lo_gap_5m` | SMA/gap | +1.11 | 0.11 | signal |
| 14 | `sma_ab_gap_5m` | SMA/gap | +1.06 | 0.09 | signal |
| 15 | `lad_50gap_30m` | SMA/gap | +0.97 | 0.12 | signal |
| 16 | `sma_bc_gap_30m` | SMA/gap | +0.83 | 0.08 | signal |
| 17 | `sma_cd_gap_5m` | SMA/gap | +0.74 | 0.08 | signal |
| 18 | `rsisma7_gap_1h` | RSI | +0.68 | 0.05 | signal |
| 19 | `bbs_pctb_5m` | Bollinger | +0.64 | 0.09 | signal |
| 20 | `rsi3_dist50_5m` | RSI | +0.62 | 0.06 | signal |

Bottom 20 permutation (smallest drop)

| Rank | Column | Family | Drop pp | SD | Flag |
|---|---|---|---|---|---|
| 161 | `rsi7_bin_1h` | RSI | -0.05 | 0.02 | signal |
| 162 | `rsi14_bin_1h` | RSI | -0.05 | 0.04 | no signal |
| 163 | `sma_ab_bin_30m` | SMA/gap | -0.05 | 0.02 | no signal |
| 164 | `htf2_cci10_bin` | CCI | -0.04 | 0.02 | signal |
| 165 | `htf1_std7_bin` | StdDev | -0.04 | 0.02 | no signal |
| 166 | `htf2_cci30_gap` | CCI | -0.03 | 0.04 | no signal |
| 167 | `sma_bc_bin_5m` | SMA/gap | -0.03 | 0.02 | no signal |
| 168 | `htf2_cci30_lvl` | CCI | -0.03 | 0.03 | no signal |
| 169 | `lad_p1_30m` | SMA/gap | -0.02 | 0.01 | signal |
| 170 | `sma_bc_bin_1h` | SMA/gap | -0.02 | 0.01 | no signal |
| 171 | `htf1_cci100_gap` | CCI | -0.02 | 0.03 | no signal |
| 172 | `htf2_std14_bin` | StdDev | -0.02 | 0.02 | no signal |
| 173 | `htf2_std14_gap` | StdDev | -0.02 | 0.02 | no signal |
| 174 | `htf1_cci100_bin` | CCI | -0.02 | 0.03 | no signal |
| 175 | `rsisma7_bin_30m` | RSI | -0.02 | 0.03 | no signal |
| 176 | `rsisma14_gap_30m` | RSI | -0.01 | 0.02 | no signal |
| 177 | `bbs_state_5m` | Bollinger | -0.01 | 0.02 | no signal |
| 178 | `ltf_cci10_bin` | CCI | -0.01 | 0.03 | no signal |
| 179 | `prev5m_dir` | prev5m_dir | -0.01 | 0.02 | no signal |
| 180 | `htf2_std7_gap` | StdDev | -0.01 | 0.02 | no signal |

Families (perm)

| Family | n | Drop pp | SD | |
|---|---|---|---|---|
| Bollinger | 32 | +8.40 | 0.12 | signal |
| SMA/gap | 68 | +7.20 | 0.17 | signal |
| HTF 30m | 55 | +4.43 | 0.12 | signal |
| HTF 1h | 55 | +3.47 | 0.11 | signal |
| RSI | 40 | +2.13 | 0.13 | signal |
| CCI | 27 | +0.20 | 0.07 | signal |
| StdDev | 12 | -0.01 | 0.04 | no signal |
| prev5m_dir | 1 | -0.02 | 0.01 | no signal |
| HTF 15m | 0 | — | — | empty |
| bar-shape | 0 | — | — | empty |
| volume | 0 | — | — | empty |

Drop-one-group retrain (Δ acc pp; negative = group was useful)

| Family | n removed | Val acc | Δ pp |
|---|---|---|---|
| SMA/gap | 68 | 57.11% | -1.16 |
| Bollinger | 32 | 58.24% | -0.03 |
| prev5m_dir | 1 | 58.26% | -0.01 |
| RSI | 40 | 58.26% | -0.01 |
| StdDev | 12 | 58.29% | +0.02 |
| CCI | 27 | 58.30% | +0.03 |

Top 10 |coef|

| Rank | Column | Family | Coef |
|---|---|---|---|
| 1 | `ltf_price_sma20_gap_5m` | SMA/gap | -2.583 |
| 2 | `bbs_sma20_gap_5m` | Bollinger | +1.111 |
| 3 | `s20_gap_5m` | SMA/gap | +1.111 |
| 4 | `bbs_mid_gap_30m` | Bollinger | -0.430 |
| 5 | `rsi7_dist50_30m` | RSI | -0.399 |
| 6 | `lad_g1_5m` | SMA/gap | -0.396 |
| 7 | `bbs_mid_gap_5m` | Bollinger | +0.387 |
| 8 | `sma_bc_gap_5m` | SMA/gap | +0.233 |
| 9 | `rsi3_dist50_5m` | RSI | -0.202 |
| 10 | `lad_50gap_5m` | SMA/gap | +0.198 |

Placebo (shuffled train y): val acc 49.69%; family mean |drop| 0.40 pp; column mean |drop| 0.30 pp.

## L4 t+2 close vs close

n=64,311 ties=1,486 purge=embargo=10m. 180 51.12% vs rule 49.66%.

Top 20 permutation

| Rank | Column | Family | Drop pp | SD | Flag |
|---|---|---|---|---|---|
| 1 | `bbs_mid_gap_30m` | Bollinger | +1.06 | 0.13 | signal |
| 2 | `s4_hi_gap_5m` | SMA/gap | +0.88 | 0.15 | signal |
| 3 | `rsi7_dist50_30m` | RSI | +0.87 | 0.15 | signal |
| 4 | `rsi_htf_cons` | RSI | +0.82 | 0.16 | signal |
| 5 | `rsisma7_gap_5m` | RSI | +0.79 | 0.29 | signal |
| 6 | `sma_ab_gap_5m` | SMA/gap | +0.70 | 0.10 | signal |
| 7 | `rsi3_dist50_30m` | RSI | +0.64 | 0.12 | signal |
| 8 | `lad_50gap_5m` | SMA/gap | +0.59 | 0.12 | signal |
| 9 | `sma_bc_gap_30m` | SMA/gap | +0.59 | 0.13 | signal |
| 10 | `rsisma14_gap_1h` | RSI | +0.58 | 0.10 | signal |
| 11 | `lad_50gap_30m` | SMA/gap | +0.56 | 0.14 | signal |
| 12 | `s714_gap_5m` | SMA/gap | +0.50 | 0.14 | signal |
| 13 | `sma_bc_gap_5m` | SMA/gap | +0.42 | 0.10 | signal |
| 14 | `s4_lo_gap_30m` | SMA/gap | +0.39 | 0.14 | signal |
| 15 | `rsi14_dist50_30m` | RSI | +0.37 | 0.12 | signal |
| 16 | `bbs_sma20_gap_5m` | Bollinger | +0.36 | 0.09 | signal |
| 17 | `rsi14_bin_30m` | RSI | +0.33 | 0.07 | signal |
| 18 | `s20_gap_5m` | SMA/gap | +0.32 | 0.05 | signal |
| 19 | `rsisma14_gap_5m` | RSI | +0.30 | 0.12 | signal |
| 20 | `lad_g1_5m` | SMA/gap | +0.29 | 0.07 | signal |

Bottom 20 permutation (smallest drop)

| Rank | Column | Family | Drop pp | SD | Flag |
|---|---|---|---|---|---|
| 161 | `ltf_std14_bin` | StdDev | -0.04 | 0.02 | no signal |
| 162 | `lad_p3_5m` | SMA/gap | -0.03 | 0.02 | no signal |
| 163 | `lad_p2_5m` | SMA/gap | -0.03 | 0.02 | no signal |
| 164 | `rbb_mid_bin_30m` | Bollinger | -0.03 | 0.01 | signal |
| 165 | `ltf_std7_bin` | StdDev | -0.02 | 0.03 | no signal |
| 166 | `s714_bin_5m` | SMA/gap | -0.02 | 0.08 | no signal |
| 167 | `htf1_std14_bin` | StdDev | -0.02 | 0.03 | no signal |
| 168 | `s4rbb_band_state_1h` | Bollinger | -0.02 | 0.05 | no signal |
| 169 | `sma_bc_cons` | SMA/gap | -0.02 | 0.02 | no signal |
| 170 | `htf2_cci100_bin` | CCI | -0.02 | 0.03 | no signal |
| 171 | `rbb_pctb_1h` | Bollinger | -0.02 | 0.02 | no signal |
| 172 | `ltf_cci100_gap` | CCI | -0.02 | 0.03 | no signal |
| 173 | `rbb_pctb_30m` | Bollinger | -0.02 | 0.06 | no signal |
| 174 | `htf1_cci10_bin` | CCI | -0.01 | 0.01 | no signal |
| 175 | `lad_p1_30m` | SMA/gap | -0.01 | 0.02 | no signal |
| 176 | `rsi7_bin_5m` | RSI | -0.01 | 0.03 | no signal |
| 177 | `htf1_cci30_bin` | CCI | -0.01 | 0.02 | no signal |
| 178 | `htf1_cci100_bin` | CCI | -0.01 | 0.03 | no signal |
| 179 | `lad_score_1h` | SMA/gap | -0.01 | 0.02 | no signal |
| 180 | `sma_ab_cons` | SMA/gap | -0.01 | 0.01 | no signal |

Families (perm)

| Family | n | Drop pp | SD | |
|---|---|---|---|---|
| RSI | 40 | +1.36 | 0.17 | signal |
| HTF 30m | 55 | +1.22 | 0.13 | signal |
| Bollinger | 32 | +0.85 | 0.19 | signal |
| CCI | 27 | +0.41 | 0.14 | signal |
| SMA/gap | 68 | +0.37 | 0.24 | no signal |
| HTF 1h | 55 | +0.09 | 0.06 | no signal |
| StdDev | 12 | +0.07 | 0.06 | no signal |
| prev5m_dir | 1 | +0.05 | 0.04 | no signal |
| HTF 15m | 0 | — | — | empty |
| bar-shape | 0 | — | — | empty |
| volume | 0 | — | — | empty |

Drop-one-group retrain (Δ acc pp; negative = group was useful)

| Family | n removed | Val acc | Δ pp |
|---|---|---|---|
| Bollinger | 32 | 50.92% | -0.20 |
| SMA/gap | 68 | 50.92% | -0.20 |
| prev5m_dir | 1 | 51.04% | -0.07 |
| StdDev | 12 | 51.07% | -0.05 |
| CCI | 27 | 51.10% | -0.02 |
| RSI | 40 | 51.19% | +0.07 |

Top 10 |coef|

| Rank | Column | Family | Coef |
|---|---|---|---|
| 1 | `ltf_price_sma20_gap_5m` | SMA/gap | +0.225 |
| 2 | `rsi7_dist50_30m` | RSI | -0.216 |
| 3 | `rsisma7_gap_5m` | RSI | -0.170 |
| 4 | `rsi14_dist50_30m` | RSI | -0.126 |
| 5 | `bbs_mid_gap_30m` | Bollinger | -0.121 |
| 6 | `lad_g1_30m` | SMA/gap | +0.120 |
| 7 | `s4_hi_gap_5m` | SMA/gap | -0.110 |
| 8 | `rsi3_dist50_30m` | RSI | -0.107 |
| 9 | `sma_bc_gap_30m` | SMA/gap | -0.103 |
| 10 | `rsi_htf_cons` | RSI | -0.101 |

Placebo (shuffled train y): val acc 50.10%; family mean |drop| 0.39 pp; column mean |drop| 0.14 pp.