# Round 3B — CEO review (IRAC)

No features implemented. No sealed exam. No merge. PR #1 untouched. After `round3-label-typ.md`. fold_60_20 only. Observation only.

## 1. Stand (one paragraph)

Round 3 asked a different question and got the same machine. On fold_60_20 val, logistic 180 on `y_5m_nextclose_vs_typ` is **61.75%** (n=65,394, ties dropped 403). The rule “this close > this typical, else 0” is **61.90%**. The model is **−0.15 pp** vs that rule, agrees with it on **94.2%** of rows, and is a coin (**48.7%**, n=3,772) when it disagrees. Logistic 128 is 61.71%. Majority 50.78%. prev5m_dir 57.27%. Placebos 49.94% / 49.91%. Calibration is honest; it is ranking the rule, not a second skill. Do not add columns. The Stage 1 ~65% was typical-price continuation through the known close-vs-typical offset. This ~62% is that same offset predicting whether the next close stays on that side of this typical. The new label still matches mapped `y_5m_dir` on **91.5%** of val rows.

## 2. Top 3 reasons this score may not be improvable at all with these data

1. **The answer is mostly already on this bar.** `close > typical` ⇔ close above (H+L)/2. EURUSD 5m next close usually sits near this close, so `close_next > TP_this` is the same side as `close_this > TP_this` (rule 61.90%). Round 3 y still equals mapped `y_5m_dir` on 91.5% of val. We changed the wording, not the information. More OHLC transforms of the same 1m stream cannot invent a next-bar surprise.

2. **X is already a pile of the same object.** Round 2 permutation: `s20_gap_5m` / `bbs_sma20_gap_5m` / `ltf_price_sma20_gap_5m` / `bbs_mid_gap_5m` are the model. 5m close-vs-mean gaps. Phase B +0.03 pp. 310 vs 180 +0.06 pp. Bar-shape (range/ATR, close-in-range, (close−typical)/ATR, wicks, tickvol) +0.06 pp. 128 vs 180 on Round 3: 61.71 vs 61.75. Extra columns that are “where did this close finish” are duplicates. They cannot beat R; they *are* R with noise.

3. **The miss is the next bar, and these files do not observe it.** Round 2 error analysis: wrong rows are next close/typical *against* this close-vs-typical (C vs R 7.4% on wrong vs 72.7% on correct). Nothing known at this close flagged them. Placebos ~50% on both labels. Consecutive 5m TP signs already flip ~41%. Tick volume is the only extra field; it was in the bar-shape probe and did nothing. No book, no flow, no next-open-as-feature (and next open ≈ this close anyway). If leftover skill exists, it is not in this CSV.

## 3. IRAC per proposed group

Verdict ∈ {keep, drop, test}. **None of these are implemented here.** Expected effect is vs the Round 3 rule (61.90%) on fold_60_20 val, unless the group is a *new question* (then vs its own majority/persist, not vs 61.90%). Confidence is prior, not a measured lift.

---

### G1 — Residual of R (new label, not a column)

**ISSUE.** 38.1% of val is where next close is *not* on this typical’s side as this close. If anything in the 180 can see a reversal, it lives here. If this slice is a coin, leftover 5m direction skill is dead.

**RULE.** A leftover skill exists only if logistic 180 on `y_res = 1{ (close_next > TP_this) ≠ (close_this > TP_this) }` beats 50% by a material margin on fold_60_20, and placebos stay ~50%. Beating 50% by 0.5 pp is noise. Do not call a coin “mean reversion.”

**APPLICATION.** Same parquet rows, same purge, seed 0, fold_60_20, drop ties on y_new or R. Do not put `y_res` in X. Mutate next close → y_res flips, X at this close unchanged.

**CONCLUSION. TEST.** Expected effect: **~50–52%**, likely a coin. Confidence **high** that it fails; that failure is the point. Cost: one label + one logistic 180. Highest information value of anything on this list.

---

### G2 — Unknown-term labels (`sign(TP_next − close_this)`, `sign(close_next − close_this)`)

**ISSUE.** Round 2 decomp: `TP_next − TP_this = (TP_next − close) + (close − TP_this)`. Known term = R. Unknown term is 80% of move variance. Close-to-close logistic already ~50.9% on this fold. If `sign(TP_next − close_this)` is also a coin, Stage 1’s 65% was only the known offset.

**RULE.** New work only if logistic 180 beats majority and persistence by a material margin on that y. Cite Round 2 close-to-close; do not pretend it is new. Placebos ~50%. Same leak test.

**APPLICATION.** fold_60_20, seed 0, 180-col X, train-only scaler. Two heads as *reference labels*, not features. Report model − 50% and model − persist. Sealed off.

**CONCLUSION. TEST** `sign(TP_next − close_this)`. **DROP** re-running close-to-close as a discovery (already scored). Expected effect: **~50–53%**. Confidence **high** it is a coin. Cost: one extra y from existing 5m OHLC.

---

### G3 — Size gate `|close − typical| / ATR` (filter, not a column)

**ISSUE.** Round 2 on `y_5m_dir`: acc rose with |TP move|/ATR (52.7% tiny → 79.5% large). That can be “R is a larger fraction of a small move” or “continuation is stronger when this bar closed far from typical.” If R still owns the large slice, the 79% is a filter on the rule, not a model.

**RULE.** A gate is useful only if (a) rule acc on the large quintile is materially above all-bar 61.90%, **and** (b) logistic 180 − rule on that slice is still ~0 or negative. If (b) is positive, then and only then is there a second skill in the tail. Do not add `|close−TP|/ATR` to X — bar-shape already did; +0.06 pp.

**APPLICATION.** Known at t. Quintiles on train |close−TP|/ATR_5m, apply frozen edges to val. Per quintile: n, rule acc, 180 acc, 180−rule, agree rate. Ablate: train-all/eval-top vs train-top/eval-top. fold_60_20, seed 0. No new columns.

**CONCLUSION. TEST** as a slice. **DROP** as a feature. Expected effect: large quintile rule **~70–80%**, 180−rule **~0 pp**. Confidence **medium-high**. Cost: groupby, no fit required for the rule table; optional one extra 180 fit on the tail.

---

### G4 — Ranker / skip-middle (product choice, not a feature)

**ISSUE.** Round 3 10-bin calibration is monotonic (actual 0.284 → 0.727). Outer bins already separate. Stage 1 freeze: outer deciles 78.92% on ~20% of bars on the *old* label. Treating every bar as a 62% call is the wrong product.

**RULE.** This is a decision, not a column. Use the **already frozen** equal-count bins. Do not search thresholds. Do not touch sealed. A skip rule is allowed as a *report* (acc @ coverage) vs rule on the same slice. Hunting a cutoff that beats the rule on val is cheating.

**APPLICATION.** Frozen Round 3 logistic 180 `P(y=1)`. Acc and 180−rule on bins 1+10 vs bins 2–9 vs all. Coverage. fold_60_20 only.

**CONCLUSION. TEST** as a product report. **DROP** as a feature or as a val-tuned threshold. Expected effect: outer ~20% **~70%+ actual rate**, middle ~50–55%; 180−rule still ~0 on every slice. Confidence **high** on ranking, **high** that it still is R. Cost: zero new fits.

---

### G5 — A question that cannot be rewritten as this close vs this typical (next 30m/1h direction, or next-bar range expansion)

**ISSUE.** Both 5m direction labels collapsed to R. If HolyGrail has any remaining job on this X, it is a *different* y: next closed 30m/1h structure (`y_30m` / `y_1h_structure` already in `labels.py`), or 1{next 5m range > this 5m range}. Comparing those accs to 61.75% is a category error.

**RULE.** New work only if logistic 180 beats persist and majority on *that* y, disagree-with-trivial-rule (if one exists) is not a coin, placebos ~50%, mutate-next does not change X. HTF y needs a longer purge (30m / 60m), not the 5m purge.

**APPLICATION.** Pick **one** first: `y_30m` on last-1m-of-5m, fold_60_20, seed 0, logistic 180 vs persist vs majority. Range expansion second if 30m is also R-like. Do not add HTF columns; 310 already failed on 5m y.

**CONCLUSION. TEST** one new question. Expected effect: **unknown**; persist on HTF is the bar to beat, not 61.90%. Confidence **low** of a win, **medium** that we learn the 180 is R-shaped on every direction head. Cost: purge change + one fit. Higher than G1–G4.

---

### G6 — Explicit close-vs-typical / bar-shape columns

**ISSUE.** The obvious “feature” is `(close−TP)/ATR`, close location in range, wicks. Round 2 already appended them. +0.06 pp. The 180 already has `ltf_price_sma20_gap_5m`, `bbs_sma20_gap_5m`, `bbs_mid_gap_5m`, `s20_gap_5m` (see §7).

**RULE.** If a candidate is a 5m close-vs-mean gap, it is a duplicate. Drop.

**APPLICATION.** Already measured. Do not repeat.

**CONCLUSION. DROP.** Expected effect: **+0.0 to +0.1 pp**. Confidence **high**. Cost wasted: we already paid it.

---

### G7 — More RSI / BB / Phase B / 310 HTF copies

**ISSUE.** “Maybe a different RSI period.” Phase B +0.03 pp val, 0 pp sealed vs 128. Round 3 128 vs 180 = −0.04 pp. 310 vs 180 = +0.06 pp. Permutation: CCI/std/`prev5m_dir` ~0; HTF almost absent from top 15.

**RULE.** A block that already lost twice is dropped. Do not retune periods.

**APPLICATION.** Already measured on both labels’ neighboring questions.

**CONCLUSION. DROP.** Expected effect: **~0 pp**. Confidence **high**.

---

### G8 — Hour / TOD / session dummies

**ISSUE.** Round 3 hour acc 59.9–63.2% (~3 pp). Round 2 old label ~6 pp. Mark: naive broker time, **no TOD/session features**.

**RULE.** Mark’s constraint stands. Even without it: a 3 pp hour spread is not a 5m direction model. Clock dummies leak session, not structure.

**APPLICATION.** Do not add. Do not “just try hour one-hots on val.”

**CONCLUSION. DROP.** Expected effect: small slice fit, not a skill. Confidence **high**. Forbidden.

---

## 4. Rank by expected value vs cost

Information EV = chance of a *decision* (kill a path or find a real leftover). Acc EV = chance of a material lift on a real question. Cost = new code + leak surface + duplicate risk.

| Rank | Group | Verdict | Acc EV | Info EV | Cost | Why this rank |
|---|---|---|---|---|---|---|
| 1 | G1 residual of R | **test** | very low | **highest** | low | Cheap kill-shot. If coin, stop 5m direction on this X. |
| 2 | G2 unknown term (TP_next−close) | **test** | very low | high | low | Explains whether 65% was only the offset. Close-to-close already dead; don’t redo it. |
| 3 | G3 size gate | **test** (slice) | none as model; maybe as filter | high | very low | Tells if 79% is R in disguise. No new columns. |
| 4 | G4 ranker / skip-middle | **test** (report) | none | medium | none | Bins already exist. Product choice, not a hunt. |
| 5 | G5 new question (30m first) | **test** | unknown, probably low | medium | medium | Only item that is not 5m-close-vs-typical. Still may be R-shaped. |
| 6 | G6 bar-shape / (close−TP)/ATR | **drop** | ~0 | none | already spent | Measured +0.06 pp. Duplicate of 180. |
| 7 | G7 RSI/BB/310 | **drop** | ~0 | none | high if repeated | Lost three times. |
| 8 | G8 TOD | **drop** | n/a | none | forbidden | Mark said no. Hour spread is small. |

Do not implement 6–8. Do not skip 1 to “add features.”

## 5. Top 5 to test first + ablation (same fold, same seed)

All five: `fold_60_20` from `scripts/round2_diagnostics.py`, seed **0**, same `X_180.parquet` rows, train-only RobustScaler on the 67 gap cols, logistic 180 C=1 max_iter=400, map 0/1 → ±1 at fit, **sealed never opened**. Labels never in X. Mutate-next-bar leak test on any new y. Observation only.

### 1. Residual of R

- `R = 1{close_this > TP_this}` (else 0). `y = y_5m_nextclose_vs_typ`. `y_res = 1{y ≠ R}` on rows where both finite.
- Fit logistic 180 on `y_res`. Report acc vs 50% and vs majority. Placebo shuffle y_res.
- **Ablation:** (A) logistic 180 vs majority vs always-0. (B) same model restricted to rows where `|close−TP|/ATR` is top vs bottom quintile. If (A) ≤ 51% and (B) still a coin, leftover skill is dead. Stop.

### 2. Unknown term

- `y_unk = sign(TP_next − close_this)`, ties dropped. **Do not** re-fit close-to-close; quote Round 2 ~50.9% as the sibling.
- Fit logistic 180 on `y_unk`. Report vs majority and vs persist (`prev5m_dir` mapped).
- **Ablation:** 180 vs 128 on `y_unk`. If 128 ≈ 180, Phase B is still dead. Placebos.

### 3. Size gate

- `s = |close_this − TP_this| / ATR_5m` (this bar, known at t). Train quintile edges → val.
- **No new columns.** Do not fit a model on `s`.
- **Ablation table:** per quintile, n / rule acc / 180 acc / 180−rule / agree rate, using the *already fitted* Round 3 logistic 180 (seed 0). Optional: refit 180 on top quintile only and compare to the frozen model on that slice (train-all vs train-tail). If 180−rule ≈ 0 in every bucket, the 79% is R.

### 4. Ranker as product choice

- Frozen Round 3 `P(y=1)` equal-count 10 bins (already in `round3-label-typ.json`). No new thresholds.
- **Ablation:** bins {1,10} vs {2–9} vs all: n, coverage, actual rate, hard acc, rule acc, 180−rule. That is the product report. Do not grid-search a cutoff. Do not score sealed.

### 5. New question (30m structure first; range expansion only if 30m is also R)

- `y_30m` already built. Last-1m-of-5m grain. Purge ≥ 30m + embargo, not 5m.
- Fit logistic 180 vs persist vs majority. Placebos. Mutate next *30m* bar, not next 5m.
- **Ablation:** 180 vs 128; 180 vs a trivial HTF analogue of R if one exists (e.g. this 30m close vs this 30m typical — **only as a baseline**, not a feature). Range expansion `1{range_next > range_this}` is a separate y; do not mix into the 30m fit.

## 6. Three considered and rejected

1. **`(close − typical) / ATR` (and the rest of the Round 2 bar-shape block).** It *is* R, scaled. Already on X in three other skins (`ltf_price_sma20_gap_5m`, `bbs_sma20_gap_5m`/`bbs_mid_gap_5m`, `s20_gap_5m`). Measured: +0.06 pp. Adding it again is cargo cult.

2. **Hour / session / TOD dummies.** Mark: no TOD. Round 3 hour spread is ~3 pp. A clock is not market structure. It would also be the first feature that is not an observation of price — it is a calendar.

3. **Another RSI/BB period set, or promoting the 310 matrix into Round 3 training.** Phase B +0.03 pp; 128 vs 180 on this label −0.04 pp; 310 +0.06 pp on the old label. Permutation already said HTF and CCI are spectators. Retuning RSI is how we waste the next week.

## 7. Constraints (enforced, shown)

**Observation only.** No entries, exits, sizing, PnL, risk, broker. Outputs are probabilities and accuracy. Stage 2 is out of scope.

**No look-ahead.** X uses 1m through **this** 5m close. y uses the **next** 5m close vs this typical. Tests already in `tests/test_round3_label.py`: shift y one 5m bar → X identical; mutate next 5m close → y flips, X at this close unchanged. Placebos 49.94% / 49.91%. Any later y (G1, G2, G5) gets the same mutate test before anyone quotes acc.

**fold_60_20 only.** Train 60% / val 20% of kept rows, newest 20% sealed **unscored**. Same windows as Round 2/3. Seed 0. Do not peek.

**No 180 duplicates.** Check against `holygrail/columns.py` `FEATURE_COLUMNS` (180 names, unique). Naive “new” columns that **must not** be added:

| Tempting add | Already in 180 | Evidence |
|---|---|---|
| close vs SMA20 | `ltf_price_sma20_gap_5m`, `ltf_price_sma20_bin_5m` | top \|coef\| #1 Round 3; perm −12.0 pp |
| close vs BB mid / SMA20 BB | `bbs_sma20_gap_5m`, `bbs_mid_gap_5m`, `bbs_pctb_5m`, `bbs_state_5m` | top \|coef\| #2, #5; perm −13.7 / −10.5 pp |
| close vs SMA20 (block 10) | `s20_gap_5m` | top \|coef\| #3; perm −13.8 pp |
| close vs SMA4 high/low | `s4_hi_gap_5m`, `s4_lo_gap_5m`, `s4_state_5m` | bar-shape sibling |
| (close−TP)/ATR, close-in-range, wicks | not named, **already probed** as bar-shape | +0.06 pp Round 2 |
| prev 5m direction | `prev5m_dir` (col 180) | baseline 57.27%; not scaled |
| RSI 3/7/14 on typical | `rsi*_bin_*`, `rsi*_dist50_*` | perm ~0 |
| RSI-SMA trio / RSI-BB | `rsisma*`, `rbb_*` | Phase B +0.03 pp |
| 1m/15m/30m/1h copies | Round 2 310 matrix | +0.06 pp vs 180 |
| hour, session, TOD | **absent** and **forbidden** | Mark; hour spread ~3 pp |

Name-collision check for the **test** list (G1–G5): none of `y_res`, `y_unk`, `y_30m`, size-quintile, or P-bins are in `FEATURE_COLUMNS`. They are labels, slices, or reports. Keep them that way.

**Duplicate-name script (do not add columns; this is the check):** `assert len(FEATURE_COLUMNS)==180 and len(set(FEATURE_COLUMNS))==180`. Proposed test ids `{y_res, y_unk, y_30m, size_q, p_bin}` ∩ set(FEATURE_COLUMNS) = ∅.

## Not done (on purpose)

No feature code. No LightGBM/MLP. No sealed scoring. No merge. PR #1 (`cursor/stage1-market-state-a365`) not touched. This file is a test plan. Do not implement from it until G1 says leftover skill exists.
