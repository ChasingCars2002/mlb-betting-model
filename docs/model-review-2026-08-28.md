# Model performance review — 2026-08-28

Follow-up to [`model-review.md`](model-review.md) (2026-08-16). That review fixed
a set of real defects and left five pieces of work undone. **The August collapse
is item D of that list coming due**, plus one fix that overshot.

---

## Executive summary

August lost **19.07% ROI** on totals — the only market still betting — over 126
graded picks (54W-72L, −20.87u on 109.46u risked). June returned +12.45% and July
+5.16% on the same market.

Three things were true at once, and only together do they explain it:

1. **The 2026-08-16 constant realignment overshot.** Correcting a projection
   that sat 0.5 runs *below* the line moved it to roughly 1 run *above*. Pick
   composition flipped from 70% Unders to 92% Overs within a day, and the Overs
   went 12-27 (−37.7% ROI).
2. **`TOTALS_SIGMA = 3.0` was ~40% below the true residual SD**, which inflated
   every edge and is why a model with no measured skill still cleared a 5% gate
   four to eight times a day.
3. **Nothing ever checked whether the projection beats the line.** It does not.
   On the 40 games with stored final scores, the projection's RMSE is 4.88
   against the posted line's 4.57, and its directional hit rate against the line
   is 12/40.

Underneath all three: the totals model had **no feedback loop**. The moneyline
side has had one since April — log the full slate, grade it, refit. Totals had
two hand-picked constants (the projection's level, and sigma) that nothing ever
compared to a realized total. Both were wrong, in both directions, for months.

**What this change does:** builds that loop, fits both constants from residuals
instead of guessing them, and gates each market on a measured out-of-sample win
over the market. Under the gate, **both markets are currently closed**. That is
the finding, not a side effect — see [What this does not do](#what-this-does-not-do).

---

## Measured evidence

### The August flip

Split on 2026-08-16, the day the score constants were realigned:

| Window | Over picks | Over W-L | Over ROI | Under picks | mean projection | mean line |
|---|---|---|---|---|---|---|
| Jun–Jul | 91 / 303 | 45-46 | −1.6% | 212 / 303 | 8.09 | 8.67 |
| Aug 17+ | 44 / 48 | 12-27 | −37.7% | 4 / 48 | 9.01 | 7.98 |

The previous review measured the pre-fix bias correctly (−0.51 runs, 67% Unders)
and corrected it by hand. Post-fix, on the games it bet, the projection sits
**+1.03 runs above the line**. The level moved past the target and out the other
side.

The specific error is not any one constant. It is that the *level* of the
projection — the thing `_BASE_RUNS`, `_LEAGUE_AVG_FIP`, `_LEAGUE_AVG_OPS` and
`_HOME_ADV` jointly determine — was being set by hand against a scale nobody had
measured. The persisted feature medians say the OPS slot actually centres on
0.721 and the FIP slot on 4.11; `score.py` normalizes them by 0.740 and 4.00.
Adjusting those constants again would have been a third guess.

### The projection does not beat the line

`actual_total` has been stored since 2026-08-18 (item 9 of the previous review).
Forty graded games is a small sample, but every measure points the same way:

| | model | posted line |
|---|---|---|
| RMSE vs actual | 4.875 | 4.568 |
| MAE vs actual | 4.218 | 3.678 |
| bias vs actual | +1.86 | +0.83 |

Correlation between the model's `projection − line` and the line's actual error
is **+0.068**. Directional hit rate against the line is **30%** (12/40).

A totals bet is a claim that the projection is closer to the truth than the line
is. On this evidence it is not, and nothing in the pipeline was checking.

### Sigma was doing the betting

With `TOTALS_SIGMA = 3.0`, a 1-run gap to the line prices as P(Over) = 0.63;
blended 50/50 against the market that is a 6.5% edge, clearing the 5% gate. At
the measured residual SD of ~4.9 the same gap prices as 0.58 — a 4.1% edge, and
no bet. Sigma alone decided whether hundreds of wagers existed.

Re-pricing August's 126 bets at higher sigmas:

| sigma | bets surviving | profit | ROI |
|---|---|---|---|
| 3.0 (shipped) | 126 | −20.87u | −19.07% |
| 4.0 | 93 | −13.59u | −14.73% |
| 4.6 | 72 | −13.59u | −17.34% |
| 5.0 | 57 | −12.04u | −17.96% |

Note what this table does *not* show: correcting sigma does not turn the month
profitable. It cuts volume, and the ROI stays around −15% to −18%. Sigma was
the mechanism, but a model that loses 17% on its most confident picks does not
have a sizing problem. That is what motivated the gate.

### Moneyline has been silent for 67 days

Last moneyline pick: **2026-06-22**. The previous review diagnosed the
mechanism (the learned blend weight rose until `(1 − w) × 0.15` fell below the
0.05 threshold, making the filter unsatisfiable) and deliberately left the
thresholds alone.

That was the right call, and the underlying reason is worth stating plainly. On
910 graded games:

| | log loss | accuracy |
|---|---|---|
| model | 0.71319 | 51.4% |
| market | 0.67907 | 55.8% |
| constant base rate | 0.69125 | — |

The classifier is worse than the market and worse than always predicting the
league home-win rate. `blend_state.json` has been carrying
`"model_adds_value": false` and `"weight_at_ceiling": true` for weeks. The
calibrator was right; nothing acted on it, and the market went quiet by
arithmetic accident rather than by decision.

---

## What changed

### 1. A feedback loop for totals

`model_log` — the full-slate, unselected log the moneyline blend is fit on — now
also records `predicted_total`, `market_total`, `over_odds`, `under_odds`, and
the realized `actual_total` at grading.

The full-slate part is load-bearing. `predictions` holds only games the filter
chose to bet, selected on the model's own disagreement with the line, which is
precisely the bias being estimated. Fitting a level correction on it would bake
that selection into the estimate. `get_graded_totals_log()` reads `model_log`
and `tests/test_totals_calibration.py` has a regression guard that plants a
99-run projection in `predictions` and asserts the fit ignores it.

One behaviour change falls out of this: the `model_prob == 0.5` "features
unavailable" fallback used to be dropped at write time. It is now written and
filtered on read, because the same game usually still carries a usable run
projection.

### 2. `totals_calibration.py`

Mirrors `calibration.py` for the totals side. After each grading run, on the
graded full slate, it fits:

- **`level_adjust`** — runs to subtract to centre the projection on realized
  totals. Applied in `score.predict_game_scores`, split evenly between the two
  teams (game-total residuals say nothing about which side is off). Clamped to
  ±2.0 runs, and a fit that hits the clamp logs an error rather than absorbing
  it: a 3-run level error is a broken projection, not an offset to trim.
- **`sigma`** — residual SD of the *centred* projection, floored at
  `TOTALS_SIGMA_MIN = 3.5`. Centred, because folding the level error into sigma
  would both understate the correction and overstate the spread.
- **`weight`** — the totals blend weight, grid-searched on log loss of the O/U
  outcome. Previously totals used the static default with no feedback at all;
  the moneyline weight was correctly *not* reused, since it is fit on a
  gradient-boosted win classifier and says nothing about a run projection.
- **`has_edge`** — whether the centred projection beats the posted line on RMSE
  against the realized total.

`TOTALS_SIGMA`'s cold-start default moves 3.0 → 4.3, and it is now only a
fallback until there is a fit.

### 3. Measured-edge gates on both markets

`config.REQUIRE_MEASURED_EDGE` (default `True`). A market emits picks only when
the model has been measured beating the market at the thing the bet depends on:
log loss of the win probability for moneyline, squared error of the projected
total for totals.

Blending toward the market caps how much damage a bad model does per bet, but it
cannot make a market-losing model +EV — it only loses the vig more slowly. Both
gates report a reason, and the moneyline gate distinguishes `no_edge` (the
classifier subtracts information — retrain) from `unreachable` (the model is
fine but the weight and threshold are inconsistent — the arithmetic problem the
last review found). They are different problems with different fixes and had
been indistinguishable from outside.

**Current state:** moneyline `no_edge` on 910 games; totals `no_calibration`
(0 of 200 graded full-slate games — the log starts empty by design, since the
only totals residuals on hand are the adversely-selected ones).

### 4. The dashboard says which market is shut, and why

`stats.json` carries `moneyline_gate`, `totals_gate`, and the totals calibration
state. An empty slate used to render "Picks post after the morning model run.
Check back later today," which is now false in a specific and misleading way —
no picks are coming until the gate reopens. It now renders the gate's reason,
and the model-status banner leads with a closed market instead of a blend
percentage.

### 5. Doubleheaders no longer grade against the wrong game

`get_yesterdays_results()` keys results by `"AWAY @ HOME"`, so on a doubleheader
date the second final game silently overwrote the first, and game 1's pick was
graded against game 2's score. ~2 doubleheaders a month reached the log. Fixed
to keep the first game deterministically and log the collision. A complete fix
needs `game_pk` recorded at prediction time; predictions carry no game number,
so the two genuinely cannot be told apart at grading.

Tests: 225 → 259, all passing.

---

## What this does not do

It does not give the model an edge, and with the gate on, **the pipeline will
emit no picks at all** until one is measured.

That is the honest state of things, not a bug to route around. Both markets have
now been measured against the market they bet into and both lose: the classifier
on 910 games, the run projection on 40. The August ROI is what betting through
that looks like. Setting `REQUIRE_MEASURED_EDGE = False` resumes betting an
unvalidated model and is not recommended.

Two paths out, in order:

**Immediate — start the totals clock.** The totals gate opens on evidence, and
there is currently none, because the full-slate totals log begins empty. At a
~13-game slate that is roughly 15 days to reach the 200-game threshold. The
pipeline collects it on its normal schedule with no intervention; nothing needs
to be bet in the meantime.

**Real — the work the previous review listed and this one did not touch.** Items
A, B, C, E and F of [`model-review.md`](model-review.md#recommended-next-work-not-in-this-change)
are unchanged and remain the only route to an actual edge:

- **A. Rebuild training features point-in-time.** Still the first thing. Holdout
  metrics are meaningless while training uses full-season aggregates to predict
  games played earlier in that season.
- **C. Add the features that actually move MLB games.** The current 15 features
  contain no team-strength term at all.
- **F. Shop lines instead of averaging them.** `fetch_live_odds()` averages
  implied probabilities across books and bets the consensus; you bet the best
  available price. On a 3.7% overround this is worth 1–2% ROI on its own —
  plausibly more than any model improvement above, and it is the one item on the
  list that does not depend on the model getting better.

**E (bet sizing) deserves a specific note.** `size_bet()` computes half-Kelly and
multiplies by `KELLY_SCALE = 13`, i.e. 6.5× full Kelly. That is only survivable
because the 3.0u cap binds. It did not cause August — the picks were losing on
their merits — but it is the reason a losing stretch on a bad model costs what it
does, and it should be fixed before either gate reopens.
