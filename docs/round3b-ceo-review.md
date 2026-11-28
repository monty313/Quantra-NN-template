# Round 3B — CEO review (IRAC)

No new features. No sealed exam. PR #1 untouched. After `round3-label-typ.md`.

## CEO paragraph

Round 3 asked a different question and got the same machine. On fold_60_20 val, logistic 180 on `y_5m_nextclose_vs_typ` is **61.75%** (n=65,394, ties dropped 403). The rule “this close > this typical, else 0” is **61.90%**. The model is **−0.15 pp** vs that rule, agrees with it on **94.2%** of rows, and is a coin (**48.7%**, n=3,772) when it disagrees. Logistic 128 is 61.71%. Majority 50.78%. prev5m_dir 57.27%. Placebos 49.94% / 49.91%. Calibration is honest; it is ranking the rule, not a second skill. Do not add columns. The Stage 1 ~65% was typical-price continuation through the known close-vs-typical offset. This ~62% is that same offset predicting whether the next close stays on that side of this typical. The new label still matches mapped `y_5m_dir` on **91.5%** of val rows.

## IRAC

### Issue

HolyGrail Stage 1 frozen logistic 180 is 64.87% sealed on `y_5m_dir` = sign(TP(next 5m) − TP(this 5m)). Round 2 showed that number is almost `sign(close − typical)` of the bar that just closed: the rule scored 65.68% vs the model 65.36%; they agreed 95.9%; when they disagreed the model was 49.4%. Wrong rows were the next close/typical going *against* this bar’s close-vs-typical. Nothing known at this close flagged the misses. Phase B RSI/BB added +0.03 pp. Combined 310 added +0.06 pp. Close-to-close was a coin (~50.9%).

Round 3 changed the question so that close-vs-typical is **not** an additive term inside y: 1 if next 5m close > this 5m typical, 0 if lower. Ties dropped. Same 180 columns, same fold_60_20, sealed unscored. The hope was leftover skill once R is no longer in the label by construction.

The result is the same copier, a bit weaker: 61.75% vs rule 61.90%, agree 94.2%, disagree = coin. 128 ≈ 180. Placebos pass. We still do not have a forecast of the next close.

### Rule

A model is doing **new work** on this X only if all of these hold:

1. Val accuracy beats the trivial rule known at t (this close vs this typical) by a **material** margin, not 0.1 pp of noise.
2. When the model **disagrees** with that rule, accuracy is clearly above 50% — otherwise the extra columns are vandalism.
3. Placebo shuffled-label and shuffled-time stay ~50% (leak check). That is necessary, not sufficient.
4. 180 vs 128, Phase B, HTF, and the 310 matrix must move the number if we claim those columns are load-bearing.

Adding features is in scope only if (1) or (2) fail **in the model’s favor**. A CEO list of ideas is a test plan, not a build list. Do not implement from this file.

A miss is normal if the next close simply does not stay on this typical’s side. A miss is a bug only if X already contains the next bar (shift-label / mutate tests) or a slice is a graveyard.

### Application

Sources: `round3-label-typ.md` / `.json`, `round2-error-analysis.md`, `round2-diagnostics.md`, `round2-results.md`, `phase-b-features.md`, `phase-c-models.md`, `stage1-strengths-freeze.md`. Fold_60_20 val. Sealed not scored.

**Round 3 (new label, same rows except new-label ties):**

| | n | Acc |
|---|---|---|
| Logistic 180 | 65,394 | **61.75%** |
| Logistic 128 | 65,394 | 61.71% |
| Rule: this close > this typical else 0 | 65,394 | **61.90%** |
| Majority (train mode = 1) | 65,394 | 50.78% |
| prev5m_dir | 64,921 | 57.27% |
| KEY: 180 − rule | | **−0.15 pp** |
| 180 agrees with rule | 61,622 | 62.55% (94.2% of rows) |
| 180 disagrees with rule | 3,772 | **48.70%** |
| Shuffled-label placebo | 65,394 | 49.94% |
| Shuffled-time 1h placebo | 65,394 | 49.91% |
| New y vs mapped `y_5m_dir` | 65,394 | 91.48% |

Logloss 180 0.6553 / 128 0.6558. Ten equal-count P(y=1) bins are monotonic (actual 0.284 → 0.727) and close to mean pred. Hour 59.9–63.2%. Weekday 61.4–62.4%. Top |coef|: `ltf_price_sma20_gap_5m`, `bbs_sma20_gap_5m`, `s20_gap_5m` — the same 5m close-vs-mean gaps Round 2 permutation named.

Leak tests: shifting y one 5m bar does not change X; mutating the next 5m close flips y and leaves X at this close unchanged. Placebos ~50%. Not a leak. Not a second skill.

**Round 2 (old label, same machine):** rule (a) 65.68% vs logistic 180 65.36%; agree 95.9%; disagree n=2,642 acc 49.4%. C vs R: wrong 7.4% vs correct 72.7%. T vs R: 5.7% vs 72.8%. 310 vs 180 +0.06 pp. Close-to-close logistic ~50.9%.

**Stage 1 / Phase B / C:** sealed logistic 180 64.87% / 0.6289, n=117,598. Persistence 58.56%. Majority 50.23%. Phase C val last-1m logistic 180 64.97% / 0.6286. Phase B +0.03 pp on val, 0 pp on sealed vs 128. LightGBM/MLP lost on val.

Why the new y is still ~R: `close > typical` iff close is above (H+L)/2. If the next close usually stays near this close (FX 5m), then `close_next > TP_this` is the same side as `close_this > TP_this`. That is 61.9% arithmetic plus stickiness, not a forecast. Overlap 91.5% with `y_5m_dir` is why 65% only fell to 62%: we barely changed the question.

(1) fails: model **loses** to the rule. (2) fails: disagree = coin. (3) holds. (4) fails: 128 ≈ 180, Phase B and 310 already dead. **Do not add features.**

### Conclusion

Stop the column hunt. 180-col logistic on both labels is a close-vs-typical readout. Stage 1 freeze stays the revert point for `y_5m_dir`. Round 3 does not replace it and does not earn a new winner. Next work is a **new question** (or a product choice to use the probability as a ranker), not another RSI/BB/HTF block.

## Top 5 to test (do not implement)

1. **Residual of R.** Label = 1 iff the next close is on the *opposite* side of this typical from this close (the 38% the rule misses). If that is a coin, there is no leftover 5m direction skill on this X.
2. **Unknown term from the Round 2 decomp.** `sign(TP_next − close_this)` and/or `sign(close_next − close_this)`. Close-to-close already ~51% on this fold. If the unknown term is also a coin, 65% was only the known offset.
3. **Size gate, not a new column.** Score only bars with large |close−typical|/ATR (Round 2 bucket 5 was 79.5% on the old label). Check whether R still owns that slice. If yes, it is a filter on the rule, not a model.
4. **Ranker as a product choice.** Outer P(y=1) bins already separate 28% vs 73% actual. Lock “use the probability, skip the middle” as a decision — or drop it — without adding features. Do not hunt thresholds on sealed.
5. **A question that cannot be rewritten as this bar’s close vs typical.** Next **30m / 1h** structure direction, or next-bar **range expansion**, on the same fold and the same leak tests. Not more 5m SMA/RSI/BB gaps.

## Not in this review

No feature code. No LightGBM/MLP. No sealed scoring. No merge. PR #1 (`cursor/stage1-market-state-a365`) not touched.
