# Model performance review — 2026-08-16

> **Superseded in part.** See [`model-review-2026-08-28.md`](model-review-2026-08-28.md).
> The totals fix in section 5 below realigned the projection's level by hand and
> overshot: picks flipped from 70% Unders to 92% Overs and August lost 19% ROI.
> Item D of the recommended work ("fit the totals model against real residuals")
> is now done, and the level is fit from data rather than set by constant.
> Items A, B, C, E and F remain outstanding and unchanged.

Diagnostic review of the MLB betting model after a stretch of poor results.
Every number below is measured from `mlb_bets.db` (804 logged games, 768 graded
picks) and the persisted model artifacts in `models/`, not estimated.

---

## Executive summary

The model has **no out-of-sample predictive skill**. On the 777 graded games it
logged for itself, its raw probability is 49.9% accurate — worse than a coin
flip, worse than always picking the home team (52.8%), and worse than a constant
prior. Its log loss (0.7198) is worse than pure market (0.6837) *and* worse than
predicting the base rate every single game (0.6915).

That is not variance. It is the downstream symptom of three upstream defects:

1. **The training matrix is mostly constant.** Over half of every fetched
   feature's training rows are the hard-coded league-average fallback.
2. **Eight of twenty-five features meant different things in training and in
   production** — a pure train/serve skew.
3. **The training features leak.** They use full-season aggregates to predict
   games played earlier in that same season.

Two more defects then converted "no edge" into "worse than no edge":

4. The moneyline pick gate has been **mathematically unsatisfiable since June**
   — the pipeline has emitted zero moneyline picks for 55 days, silently.
5. The totals score model is **off-scale by ~11%**, projecting ~0.5 runs under
   the market and making 67% of all O/U picks Unders.

Bottom line: **the model should not be staked until it can beat a constant
prior out of sample.** The blend weight already knows this — it has been pinned
at its 0.95 ceiling, meaning the unconstrained optimum is "ignore the model".

---

## Measured evidence

### The model has negative information

777 graded games from `model_log` (full slate, no adverse selection):

| | Model | Market | Constant prior |
|---|---|---|---|
| Log loss | **0.7198** | 0.6837 | 0.6915 |
| Brier | **0.2624** | 0.2454 | — |
| Accuracy | **49.9%** | 54.3% | 52.8% |
| Prob. std. dev. | 0.126 | 0.072 | — |

The model is *more* dispersed than the market (0.126 vs 0.072) while being less
accurate: it is confidently wrong. Its reliability curve is nearly flat and
partly inverted — the games it likes least win more often than the games it
likes most in the bottom half of the range:

| Model says | Actually won | Market said |
|---|---|---|
| 0.318 | 0.505 | 0.474 |
| 0.402 | 0.433 | 0.497 |
| 0.462 | **0.660** | 0.516 |
| 0.515 | 0.546 | 0.527 |
| 0.561 | 0.443 | 0.536 |
| 0.604 | 0.546 | 0.541 |
| 0.659 | 0.526 | 0.550 |
| 0.716 | 0.561 | 0.580 |

A useful model's right column rises with its left. This one barely moves.

For contrast, the **market** probabilities in the same table are well
calibrated (0.417→0.442, 0.481→0.496, 0.544→0.512, 0.634→0.636), which confirms
the odds pipeline is attaching the right prices to the right games. The problem
is the model, not the market data.

### The realized book

| Market | Record | Units | Profit | ROI |
|---|---|---|---|---|
| Moneyline | 165–191 | 476.5 | +18.87 | +3.96% |
| Totals | 204–192 | 368.6 | −4.75 | −1.29% |
| **Combined** | **369–383** | **845.1** | **+14.12** | **+1.67%** |

The moneyline ROI is *not* evidence of edge — it comes from 356 bets whose
underlying probabilities are shown above to be noise, and it is well inside the
standard error. It is also **frozen**: the last moneyline pick was 2026-06-22.

---

## Defects, in order of impact

### 1. The training matrix is mostly league-average constants — CRITICAL

`models/feature_medians.joblib`, written by the last training run, contains:

```
home_p_xFIP_season   4.2000     home_bullpen_era   4.0000
home_p_SIERA_season  4.2000     home_bullpen_fip   4.0000
home_p_K_BB_pct_...  10.0000    home_hit_wrc_plus  100.0000
home_p_WHIP_season   1.3000     home_hit_ops       0.7400
...
```

Every one of those is **exactly** the hard-coded fallback constant in
`data.py` (`_default_pitcher_stats()`, `get_bullpen_stats()`,
`get_team_hitting_splits()`). A median lands exactly on the fallback only if at
least half the rows *are* the fallback. This holds for all 24 fetched features
simultaneously — `park_factor` (0.99, from a static table) is the only feature
with a real distribution.

Note that `xFIP_season` and `SIERA_season` are computed from different inputs
(FIP vs. ERA) and would essentially never be equal on real data, let alone both
be exactly 4.20.

The cause is structural: every fetch in `data.py` catches its own exception and
returns a league-average constant. That is correct for one missing pitcher on a
live slate, but during a bulk training build a failing endpoint quietly produces
a full-size, signal-free matrix that trains and saves without complaint.
`train_models()` then reports holdout metrics on that matrix and the pipeline
carries on.

**Fixed:** `features.check_feature_quality()` now measures and logs the
fallback rate per feature, and `train.run_incremental_retrain()` aborts rather
than overwriting a live model when any feature exceeds 25% fallbacks
(`--allow-degraded-data` overrides). The abort returns a failure status that
the CLI turns into a non-zero exit — both `weekly-retrain.yml` and the
`|| python main.py --retrain` fallback in `daily-predict.yml` read the exit
code, so a silent abort would have reported a green retrain that never ran and
left the daily run predicting with a stale, schema-mismatched model.

**Still to do:** find out *why* the fetches fail. The most likely culprit is
`get_historical_game_data()`, which relies on `hydrate=probablePitcher` for
*completed* seasons; MLB's schedule endpoint does not reliably populate
`probablePitcher` on final games, so `home_pitcher_id` comes back `None` and
`get_pitcher_stats()` returns defaults for every row. If so, pull the actual
starter from each game's boxscore/linescore rather than the probable-pitcher
hydrate. **Run a retrain and read the new fallback-rate log lines before
trusting any model this repo produces.**

### 2. Train/serve skew on eight features — CRITICAL

`features.build_training_features()` calls `get_pitcher_stats(...,
use_rolling=False)`, and that flag makes `data.py` copy each season value
straight into its rolling slot:

```python
# data.get_pitcher_stats, training path
rolling_stats = {
    "xFIP_rolling": season_stats["xFIP_season"],   # identical, every row
    ...
}
```

At prediction time `use_rolling=True` fetches a real 30-day window. So eight of
twenty-five model inputs were **exact duplicates of another column during
training and carried genuinely different values in production**. The trees split
arbitrarily between each duplicated pair — feature importance is near-uniform at
~1/25 per column, the signature of a model finding nothing — and every one of
those splits was then evaluated on a shifted distribution live.

Two more columns, `home_hit_wrc_plus` / `away_hit_wrc_plus`, are perfectly
collinear with their OPS counterparts by construction
(`wrc_plus = 100 * ops / 0.720`).

**Fixed:** both groups removed from `FEATURE_COLUMNS` (25 → 15 features). The
schema hash change forces a full rebuild on the next retrain.

**Still to do:** recent form is real signal and it should come back — but only
once the *training* path can build genuine point-in-time windows. See
"Recommended next work" below.

### 3. Look-ahead leakage in training features — CRITICAL

`build_training_features()` fetches season-level stats for the whole season and
applies them to every game in it:

```python
pitcher_cache[hp_key] = get_pitcher_stats(hp_id, hp_name, season=season, ...)
bullpen_cache[hb_key] = get_bullpen_stats(game["home_team"], season=season)
hitting_cache[hh_key] = get_team_hitting_splits(..., season=season)
```

A game played on 2024-04-05 is therefore predicted using the pitcher's *final*
2024 ERA, the team's *final* 2024 bullpen ERA, and the team's *final* 2024
platoon OPS — all of which incorporate the outcome of that very game and every
game after it. Holdout metrics computed on this matrix are meaningless, which is
exactly why offline numbers looked acceptable while live performance is at
chance.

This is the single largest remaining item and is **not fixed in this change** —
see "Recommended next work".

### 4. The moneyline gate has been unsatisfiable since June — HIGH

Because every probability is shrunk toward the market before the edge is
measured, a pick's edge is capped at `(1 - weight) * MAX_RAW_DISAGREEMENT`.
With `EV_THRESHOLD = 0.05` and a 0.15 cap, the filter becomes impossible to
satisfy once `weight > 0.667`.

The self-tuned weight has been pinned at its **0.95** ceiling since calibration
began: max achievable edge `0.05 × 0.15 = 0.0075`, against a required 0.05 and
a ~3.7% market overround. Simulating the live filter over all 777 logged games:

| Blend weight | Picks generated | ROI |
|---|---|---|
| 0.30 | 268 | −3.26% |
| 0.40 | 224 | +1.23% |
| 0.50 | 153 | −0.77% |
| 0.60 | 79 | −7.93% |
| **0.667+** | **0** | — |
| **0.95 (live)** | **0** | — |

At 0.95 the count is zero even with the threshold set to 0.0 — the shrunk
probability can never clear the vig. The database agrees: **zero moneyline picks
since 2026-06-22.**

Note the deeper point: at every weight that *does* produce picks, ROI is
approximately zero or negative. So emitting nothing is the *right* answer for a
model with no edge. The bug is not the silence — it is that the silence was
**invisible**, reported as an ordinary "0 picks today", while the dashboard kept
advertising a frozen +3.96% moneyline ROI as though the strategy were live.

A related consequence: `compute_confidence()` computes
`span = max_edge - EV_THRESHOLD`, which goes negative in this regime and made
the function return 1 star for every pick.

**Fixed:** `evaluate.edge_band()` reports reachability; the daily run prints and
logs a warning when the gate is unreachable; the state is exported to
`stats.json`; `compute_confidence()` now ranks against the achievable band
instead of collapsing to a flat 1.

**Deliberately not changed:** the thresholds themselves. Loosening them to force
picks out of a model with no demonstrated edge would convert a harmless silence
into an active losing strategy. That is a call for the owner to make once the
model can beat a constant prior.

### 5. The totals score model is off-scale — HIGH

`score.py` divided the `xFIP` feature slot by `_LEAGUE_AVG_ERA = 4.50`. But that
slot is not xFIP and not ERA: `data._compute_fip()` builds it as FIP with
`_FIP_CONSTANT = 3.10`, whose league mean is ~4.00. Dividing a ~4.00-scale
quantity by 4.50 multiplied every projection by ~0.89.

Measured over 396 graded totals picks:

- mean projected total **8.10** vs mean posted line **8.62** (−0.51 runs)
- median gap **−1.06** runs
- the model sat below the line on **67.4%** of games
- resulting picks: **267 Unders vs 129 Overs**
- Over-side ROI: **−19.08%** (adverse selection — the only Overs that cleared
  the gate were those where a downward-biased estimator still exceeded the line)

Two secondary errors compounded it: the `_LEAGUE_AVG_OPS = 0.720` reference did
not match the 0.740 that `get_team_hitting_splits()` actually returns as its
league default, and the projection attributed **100% of run prevention to the
starting pitcher**, ignoring the bullpen features entirely even though relievers
throw ~45% of innings. That left the projection swinging ~1.4 runs around lines
that books move by ~0.4.

**Fixed:** reference constants realigned to the scale of the features they
normalize; bullpen blended in by innings share; response damped so extreme
inputs stay in a plausible band. A league-average matchup now projects 8.79
rather than 8.10.

### 6. Odds could be matched to the wrong game in a series — MEDIUM

`match_odds_to_games()` keyed its lookup on `(home_team, away_team)` only:

```python
for o in odds:
    odds_lookup[(o["home_team"], o["away_team"])] = o   # last event wins
```

The-Odds-API returns every upcoming event, and MLB plays the same matchup on
3–4 consecutive days. The last event for a matchup won the dict slot, so
today's game could be priced off a later game in the same series — different
starting pitcher, different line. `commence_time` was fetched and then never
used.

I could not confirm this fires in production (the logged market probabilities
are well calibrated, which argues the API's return window is short), so treat it
as a latent bug rather than a proven cause. It is cheap to close either way.

**Fixed:** lookup keyed on `(home, away, commence_date)` with Eastern-time
bucketing (a 10pm ET first pitch is the *next* day in UTC), earliest start
winning a doubleheader, and a team-pair fallback when `commence_time` is absent.

### 7. Totals prices averaged across different lines — MEDIUM

`fetch_live_odds()` collected every book's Over price into one list, every Under
price into another, and the posted lines into a third — then averaged all three
independently. If book A posts 8.5 and book B posts 9.0, the result is a price
pair belonging to *neither* line, attached to an averaged 8.75 line that nobody
offered.

**Fixed:** prices are bucketed by line; the most widely posted line wins, and
only prices quoted at that line are averaged. `num_books_at_line` is exported.

### 8. Every totals push was booked as a full-unit decision — MEDIUM

`grade_predictions()` computed `"Over" if actual_total > listed else "Under"`.
On a whole-number line (9.0) landing exactly on the total, the stake is
returned — but this graded it as an Under, booking a **full-unit loss on every
Over that pushed and a full-unit win on every Under that pushed**. With 267
Under picks in the book, this materially distorts the recorded totals record in
the Unders' favour.

**Fixed:** pushes grade to `status = "Push"`, `profit = 0.0`, and are excluded
from the win/loss record and the ROI denominator — server-side in
`get_roi_stats()` and in `docs/js/dashboard.js`, which computes its own stats
from the exported history and previously mapped only Win/Loss/Pending (an
unknown status fell through to pending styling and vanished from the record).

### 9. Final scores were never stored — MEDIUM

`grade_predictions()` computed `actual_total` and threw it away. That makes the
totals model impossible to calibrate: `TOTALS_SIGMA`, the run-scale constants,
and the residual centering all need real residuals to fit against.

**Fixed:** `actual_home_score`, `actual_away_score`, and `actual_total` are now
persisted at grading time and exported.

### 10. Headline stats reported a market the model had stopped betting — MEDIUM

`get_roi_stats()` filtered to `bet_type = 'moneyline'`, but `export_dashboard_data()`
published the result under the generic keys `ytd` and `all_time`. Since the
moneyline gate closed in June, the dashboard's headline has been a frozen
snapshot of a strategy that no longer runs, while the *only* market actually
being bet (totals, −1.29%) was invisible.

The pending count inside the same function ignored the `bet_type` filter
entirely, so a moneyline-only block reported every market's pending picks.

**Fixed:** `get_roi_stats(bet_type=...)` scopes both the graded and pending
queries consistently; `ytd`/`all_time` now cover all markets; a `by_bet_type`
breakdown and the model-health flags are exported.

### 11. Schema drift would have been silently zero-filled — LOW

`predict_win_prob()` did `X.reindex(columns=model_cols, fill_value=0.0)`. Since
0.0 is a *plausible-looking* value for every feature here — an ERA, FIP, or
WHIP of 0.0 reads as an unhittable pitcher — a mismatch between a saved model
and `FEATURE_COLUMNS` would produce confident nonsense rather than an error.

**Fixed:** missing columns are imputed from training medians and logged at
ERROR.

### 12. Calibration failure had no alarm — LOW

`blend_state.json` records `log_loss: 0.68419` against
`pure_market_log_loss: 0.68373`. The best achievable blend is **worse than
ignoring the model entirely**, and the chosen weight sits exactly on the
`BLEND_WEIGHT_MAX` ceiling — meaning the grid search was cut off, not satisfied.
Both facts were recorded in the state file and never acted on or surfaced.

**Fixed:** `update_blend_weight()` now emits `weight_at_ceiling` and
`model_adds_value`, logs at ERROR when blending cannot beat the market, and
exports both to the dashboard.

---

## Recommended next work (not in this change)

These are the items that actually restore an edge. They are ordered by expected
value, and none of them is a config tweak.

### A. Rebuild training features point-in-time — do this first

Nothing else matters until the training matrix reflects what was knowable
*before* first pitch. Concretely:

- For each game date, fetch stats over `[season_start, game_date)` rather than
  the whole season. `data._mlb_api_get` already speaks `byDateRange`; the
  pitching leaderboard endpoint accepts the same window.
- Cache one league-wide leaderboard **per date** (~180 calls/season, ~720 total)
  instead of one per season. Team pitching and hitting splits can be pulled the
  same way.
- Guard the early season: with fewer than ~40 IP, regress the pitcher toward
  league average (or toward prior-season performance) rather than using a
  10-inning ERA at face value.
- Once this exists, restore the `*_rolling` columns — they will finally mean
  the same thing in training and production.

Expect offline metrics to get **worse**, and that is the point: the current
numbers are inflated by leakage. A leak-free holdout log loss below 0.687 is the
first real evidence of skill.

### B. Validate the way the model is actually used

Replace the random-ish `CalibratedClassifierCV(cv=5)` on temporal data with
walk-forward validation: train through date *T*, predict *T+1*, roll forward.
Score it against the two benchmarks that matter — the constant home-win prior
and the de-vigged market — not just against itself. If it cannot beat the
constant prior out of sample, it is not ready to stake, full stop.

### C. Add the features that actually move MLB games

The current 15 inputs contain no measure of overall team strength. Highest-value
additions, roughly in order:

- **Team strength** — season-to-date run differential or a Pythagorean win
  expectation for each side. This is the single biggest omission.
- **Starter workload context** — days of rest, times through the order, recent
  pitch counts. `get_pitcher_days_rest()` already exists in `data.py` and is
  **never called by anything**.
- **True bullpen split** — `get_bullpen_stats()` currently returns the whole
  staff's ERA (its own docstring admits this), so the starter's contribution is
  double-counted on both sides. Use the relievers-only split.
- **Bullpen fatigue** — relief innings thrown in the last 3 days.
- **Lineup quality** — the posted lineup, or at least a flag for missing
  regulars. Team-level platoon OPS says nothing about who is playing today.
- **Weather** — temperature and wind, which matter enormously for totals.
- **Travel/schedule** — getaway days, time-zone changes, consecutive games.

`BULLPEN_ROLLING_DAYS` is defined in `config.py`, accepted as a parameter by
`get_bullpen_stats()`, and then never used — the function always returns
season-to-date. Either implement it or drop it.

### D. Fit the totals model against real residuals

Now that `actual_total` is stored, after ~200 graded games:

- Regress `actual_total` on `predicted_total` to recenter `_BASE_RUNS` and check
  the slope (a slope well under 1 means the projection is still over-dispersed
  and the elasticity constants need lowering).
- Fit `TOTALS_SIGMA` from the residual standard deviation. The current **3.0 is
  almost certainly too low** — the residual SD around a good MLB total
  projection is nearer 4.0–4.3. Too small a sigma inflates P(Over)/P(Under),
  overstates every edge, and oversizes every Kelly stake. I left it at 3.0
  rather than substituting another guess; fit it from the data.

### E. Fix bet sizing

`size_bet()` computes half-Kelly and then multiplies by `KELLY_SCALE = 13`.
That is not half-Kelly — it is 6.5× full Kelly, a bankroll-destroying fraction
if it ever binds — and in practice it *does* bind: 53 bets landed on the 3.0u
cap while 117 sat on the 0.5u floor. Between the arbitrary 13× multiplier and
the clamps, the sizing carries almost no information about the actual edge. Pick
a real fractional-Kelly (0.25–0.5×), express units as a genuine percentage of
bankroll, and let the cap be a risk limit rather than the primary mechanism.

### F. Shop lines instead of averaging them

`fetch_live_odds()` averages implied probabilities across all books and bets the
consensus. You bet at the **best available price**, not the average one. Use the
consensus (ideally de-vigged per-book, then averaged) as the *fair-value*
estimate, and the best available price for EV and staking. On a 3.7% overround
this is worth roughly 1–2% ROI on its own — plausibly more than any model
improvement listed above.

---

## What changed in this PR

| File | Change |
|---|---|
| `score.py` | Reference constants realigned to feature scale; bullpen blended in by innings share; damped response |
| `features.py` | Removed 8 skewed `*_rolling` + 2 collinear `*_wrc_plus` columns; shared row builder; `check_feature_quality()` |
| `train.py` | Aborts retrain on degraded feature quality; `--allow-degraded-data` override |
| `database.py` | Push grading; final scores persisted; `get_roi_stats(bet_type=...)` with consistent pending scoping |
| `odds.py` | Date-scoped odds matching; totals prices bucketed by line |
| `evaluate.py` | `edge_band()`; confidence no longer collapses to 1 star; pushes in stats output |
| `calibration.py` | `weight_at_ceiling` / `model_adds_value` health flags with ERROR logging |
| `main.py` | Warns when the pick gate is unreachable; exports per-market stats and health flags |
| `model.py` | Schema drift imputes medians and logs instead of silently zero-filling |
| `docs/js/dashboard.js`, `docs/css/style.css` | Render and account for the new `Push` status |
| `tests/` | 48 new regression tests (177 → 225) |

No betting thresholds were loosened. The changes fix defects and make failure
states visible; they do **not** by themselves give the model an edge. Items A–C
above are what does that.
