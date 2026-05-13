# MLB Betting Model — Logic Review

Date: 2026-05-13
Branch: `claude/review-model-logic-zRtdR`

This review covers the data pipeline (`data.py`), feature engineering
(`features.py`), training (`train.py`, `model.py`), inference (`model.py`),
score model (`score.py`), and bet selection (`evaluate.py`, `odds.py`).

Findings are ranked by impact on real-money performance. Severity:
**Critical** = directly biases bets toward losers or causes ruin-level
sizing; **High** = degrades model accuracy materially; **Medium** = leaks
or correctness bugs with bounded impact; **Low** = cleanup.

---

## 1. Training-time look-ahead bias (Critical)

**Location:** `features.build_training_features`, `data.get_pitcher_stats`,
`data.get_bullpen_stats`, `data.get_team_hitting_splits`.

When training on a 2024 April game, the feature pipeline calls
`pybaseball.pitching_stats(2024, 2024)` and uses the pitcher's
**end-of-season** xFIP, SIERA, K-BB%, and WHIP as features. The same is
true for bullpen ERA/FIP and team hitting splits — all values are
full-season aggregates that include data from after the game date.

This is direct target leakage: an April start uses the pitcher's
August–September performance to predict the April outcome. The model
appears to do well on holdout, but at inference the live values it sees
are point-in-time and have a very different distribution from training.

**Why this matters most.** Every reported metric (Brier, log-loss, ECE,
backtested ROI from the training split) overstates real performance,
because training rows include information that is not available at the
moment the bet is placed.

**Fix.** Replace season-aggregate fetches with point-in-time (PIT)
queries:

- For pitchers: `pybaseball.pitching_stats_range(season_start, day_before_game)`
  for each unique `(pitcher_id, game_date)` pair. Cache by
  `(pitcher_id, season, asof_date)` — typical season has ~200 unique
  starts per starter, so the cache is small.
- For bullpens and hitting: same approach with team-level date ranges.
- For very early-season starts (e.g. first 2 weeks), fall back to
  prior-season stats or league average rather than the current-season
  partial — and store an `as_of` indicator the model can learn from.

This is a refactor (~few hundred lines of new caching + IO), but
without it none of the downstream metrics can be trusted.

---

## 2. Train/predict feature distribution mismatch (Critical)

**Location:** `features.build_training_features` lines 245-248,
`data.get_pitcher_stats` with `use_rolling=False`.

Several features that vary at inference are **constants** during
training:

| Feature                  | Training value          | Inference value          |
|--------------------------|-------------------------|--------------------------|
| `home_p_days_rest`       | constant `5.0`          | live MLB Stats API value |
| `away_p_days_rest`       | constant `5.0`          | live MLB Stats API value |
| `home_team_ops_10`       | constant `0.735`        | live last-10-game OPS    |
| `away_team_ops_10`       | constant `0.735`        | live last-10-game OPS    |
| `*_p_xFIP_rolling`       | copy of `*_xFIP_season` | true 30-day window       |
| `*_p_SIERA_rolling`      | copy of `*_SIERA_season`| true 30-day window       |
| `*_p_K_BB_pct_rolling`   | copy of season          | true 30-day window       |
| `*_p_WHIP_rolling`       | copy of season          | true 30-day window       |

Consequences:
- The four "rolling" pitcher columns are perfectly collinear with their
  season counterparts during training. The model effectively learns
  4 pitcher features, not 8. Tree splits on `_rolling` columns are
  redundant with `_season`.
- `*_days_rest` and `*_team_ops_10` have **zero variance** during
  training, so trees never split on them and the calibrated probability
  ignores them — even though they vary at inference and may carry real
  signal.
- At inference the rolling features can drift far from season values
  (e.g., a slumping pitcher with rolling xFIP 5.5 and season xFIP 3.8).
  The model has no learned response to that divergence and will simply
  re-use the season weight, producing distorted probabilities.

**Fix.** Either (a) compute the rolling/derived features point-in-time
during training so they actually vary, or (b) drop the redundant
inference-only features from `FEATURE_COLUMNS` entirely. Option (b) is
honest and trivially actionable; option (a) is correct and requires the
same PIT infrastructure as item 1.

---

## 3. Calibration data leakage via random k-fold (Critical)

**Location:** `model.train_models` — every `CalibratedClassifierCV(..., cv=5, method=...)`.

`CalibratedClassifierCV` defaults to a **stratified random KFold**.
Applied to time-ordered MLB games, this means each calibration fold sees
games from across all seasons. Probabilities for an early-season 2023
game are calibrated using games played in late 2025.

Effects:
- The reported Brier/log-loss/ECE on the temporal holdout still look
  optimistic, because the calibrator (fit on the 80% train half) has
  already absorbed structure that wraps around the holdout boundary
  through its random folds.
- Live predictions are calibrated against a future-leaking distribution
  that does not match what we will see going forward.

**Fix.** Pass `cv=TimeSeriesSplit(n_splits=5)` explicitly to every
`CalibratedClassifierCV` call. Implemented in this change.

---

## 4. EV is mathematically correct, but `edge` is computed against the vigged line (High)

**Location:** `evaluate.calculate_edge`, `evaluate.filter_positive_ev`,
`odds.fetch_live_odds`.

`american_to_implied_prob` converts each side's odds independently. For
a typical -110/-110 market the two implied probabilities sum to ~1.048,
i.e. the line carries ~4.8% vig. `calculate_edge(model_prob, implied)`
then measures the model against the **vigged** line, not the fair line.

Practical impact:
- The reported "Edge" column in picks is systematically understated by
  roughly half the vig (~2.4% on -110 markets, more on bigger dogs).
- `EV_THRESHOLD = 0.02` is being applied to a quantity that is already
  ~2.4% below the true edge against fair value, so the threshold is
  effectively much stricter than intended on some markets and far too
  loose on others (it scales with vig per-game).

Note that **EV itself is computed correctly** — it uses
`model_prob * (decimal_odds - 1) - (1 - model_prob)` which is the
breakeven-from-odds formulation and ignores `implied_prob` entirely. So
the `home_ev > 0` filter is sound. The bug is purely in `edge` reporting
and in the `edge >= EV_THRESHOLD` second condition.

**Fix.** Add `devig_pair(home_odds, away_odds)` (proportional / shin
normalization) and use the de-vigged "fair" probabilities for `edge`
calculation and display. Keep `home_ev > 0` as the binding
profitability filter and treat the edge threshold as a confidence
floor. Implemented (proportional method) in this change.

---

## 5. Kelly stake scaling is catastrophic (Critical)

**Location:** `evaluate.size_bet`, `config.KELLY_SCALE = 13`.

```python
kelly = (model_prob * (decimal_odds - 1) - (1 - model_prob)) / (decimal_odds - 1)
half_kelly = kelly * 0.5
return ... * KELLY_SCALE  # KELLY_SCALE = 13
```

`half_kelly * 13` is **6.5× full-Kelly**. The literature is unambiguous
that anything above ~1× Kelly has positive risk of ruin even with a
correct edge; classical full-Kelly already has 50% drawdown probability
of ~0.5. At 6.5× Kelly on a model with the calibration issues above,
ruin is essentially guaranteed.

The capping at `MAX_BET_UNITS = 3.0` masks part of this on small edges,
but any 4-5%+ edge clamps the size right to 3 units, which on a 100-unit
bankroll is 3% — already aggressive when stacked across 5-10 picks/day
that are correlated through league-wide variance.

**Fix.** Set `KELLY_SCALE = 1.0` so the function returns true
half-Kelly. Keep the `MAX_BET_UNITS = 3.0` cap as a safety net. The user
can scale up if they decide they trust the model. Implemented.

---

## 6. Park factor applied only to home runs (High)

**Location:** `score.predict_game_scores`.

```python
home_runs = base * (home_xfip / lg_era) * (home_ops / lg_ops) * park * HOME_ADV
away_runs = base * (away_xfip / lg_era) * (away_ops / lg_ops)
```

Park factor is multiplied into `home_runs` only. But the park is the
**venue**, not a team trait — the away team also scores in that park.
Coors Field (park factor 1.22) inflates both teams' runs, not just the
home team's.

**Fix.** Multiply `park` into both `home_runs` and `away_runs`.
Implemented.

---

## 7. Bullpen reliever heuristic uses OR where AND is needed (High)

**Location:** `data.get_bullpen_stats`.

```python
relievers = pitching[
    (pitching["GS"] < 5) | (pitching["G"] - pitching["GS"] > 10)
]
```

This admits any pitcher who either has few starts OR has >10 relief
appearances. A swingman with 22 GS and 30 G satisfies the second
condition and is counted in the bullpen, even though they're primarily
a starter. Their ERA dilutes the bullpen aggregate.

**Fix.** Use `&` so a pitcher must have few starts AND substantial relief
work to count as a reliever. Implemented.

---

## 8. "Platoon adjustment" doesn't actually use L/R splits (High)

**Location:** `data.get_team_hitting_splits`.

The function is named "splits" and accepts `vs_hand`, but it fetches
**aggregate** team wRC+ and OPS across all opponents and then multiplies
by a constant fudge factor:

```python
if vs_hand == "L":
    wrc_plus *= 0.97; ops *= 0.97
else:
    wrc_plus *= 1.03; ops *= 1.03
```

The same scalar is applied to every team. The Yankees' true OPS vs LHP
might be 0.820 with their season aggregate at 0.760, while another team
may be 0.700 vs LHP and 0.780 overall. The current code maps both to
`0.760 * 0.97`, erasing all team-level platoon signal.

This also has the side effect that lefty-heavy lineups (which mash RHP)
are systematically *underrated* against L, and righty-heavy lineups are
*overrated* against L.

**Fix.** Use pybaseball / FanGraphs splits leaderboards by handedness
(`fg_team_batting_data` with a `vs_hand` filter, or query the splits
endpoint directly). For now, the code at minimum should be documented
as a placeholder. This is left as a documented TODO in code comments;
the data work is non-trivial.

---

## 9. `predict_win_prob`: feature reindex can silently zero out features (Medium)

**Location:** `model.predict_win_prob` lines 349-369.

When the model's stored feature list differs from the current
`FEATURE_COLUMNS` (e.g., features added after the model was saved), the
code does `X.reindex(columns=model_cols, fill_value=0.0)`. A new feature
silently becomes 0 instead of triggering a retrain.

**Fix.** Compare `set(model_cols)` against `set(FEATURE_COLUMNS)` and
log a `warning` (or raise) when they diverge. The training-state hash
in `train.py` already exists for this purpose — wire it into the
inference path too.

---

## 10. `is_home` removed but home-field advantage now has no explicit feature (Medium)

**Location:** `features.FEATURE_COLUMNS`.

The comment says `is_home` was excluded because it was always 1. That's
correct for a single-row prediction, but during training every game has
both a "home team" view (`is_home=1`) and an implicit away view; the
target is `home_win`. So `is_home` was always 1 by construction.

However, home-field advantage in MLB is real (~3-4% baseline). The
model does not currently have a feature that encodes it — `park_factor`
is a venue offensive characteristic, not a generic home edge. The
constant prior gets absorbed into the model intercept, which is fine
**unless** the home-team distribution differs across seasons/parks,
which it doesn't. Net: probably ignorable. Documented for completeness.

---

## 11. `_evaluate_model` log_loss can throw or produce -inf on extreme probabilities (Low)

**Location:** `model._evaluate_model`.

`log_loss(y_true, y_prob)` on probabilities that contain 0.0 or 1.0
exactly will return inf. sklearn does clip internally to `1e-15`, so in
practice this is fine — but it's worth clipping explicitly for safety
when probabilities come from XGBoost without calibration. Tightened
explicitly in this change.

---

## 12. Discord/GitHub formatting trusts `p["odds"]` is int (Low)

**Location:** `main.post_picks_to_discord`, `main.post_picks_to_github_issue`.

`p["odds"] > 0` will crash if odds is ever None. Database round-trip can
return None when the column is null. Wrap in `isinstance(odds, (int, float))`
check. Not implemented here — low impact, not on hot path for the model.

---

## 13. `_compute_ece` last-bin boundary (Low)

**Location:** `model._compute_ece`.

Final bin uses `<=` (inclusive) while others use `<` (exclusive). A
probability of exactly `0.1` will fall in bin 1, but the bin-9 to
bin-10 boundary is handled correctly with the special case. Standard
ECE convention. No fix needed.

---

## 14. Pitcher hand cache initialization missing for first-of-season starters (Low)

**Location:** `features.build_training_features` line 173-174.

```python
home_pitcher_hand = hand_cache.get(hp_id, "R")
```

When `hp_id` is NaN (no probable pitcher recorded for a historical
game) we fall back to "R". This is a small bias because LHP are ~30%
of MLB starters; for those games the model trains with the wrong
opposing-pitcher-hand split assumption. Reasonable to leave — the
"hitting splits" function fudge factor (item 8) dwarfs this anyway, and
once item 8 is fixed properly this becomes more material.

---

## 15. `get_team_rolling_ops` per-game OPS extraction (Medium)

**Location:** `data.get_team_rolling_ops`.

The function reads `stat.ops` from each gameLog split. The MLB Stats
API `gameLog` group returns **per-game** stats in each split — so this
is correct. However, some splits' `stat.ops` is the cumulative season
OPS up to that point, depending on the endpoint variant. Spot-check
against a known game (e.g., team 147 / NYY on a specific 2024 date) is
worth doing. Not changed in this pass.

---

## Summary of changes applied on this branch

| # | Severity  | Fix                                                                  | Status                  |
|---|-----------|----------------------------------------------------------------------|-------------------------|
| 3 | Critical  | `CalibratedClassifierCV` switched to `TimeSeriesSplit(n_splits=5)`   | Applied                 |
| 4 | High      | Added `devig_pair` + use vig-removed fair odds for `edge` and `EV`   | Applied (proportional)  |
| 5 | Critical  | `KELLY_SCALE` lowered from 13 → 1.0 (true half-Kelly)                | Applied                 |
| 6 | High      | Park factor applied to both teams in score model                     | Applied                 |
| 7 | High      | Bullpen reliever filter: `&` instead of `|`                          | Applied                 |
| 8 | High      | Documented platoon-split placeholder in code (data work outstanding) | Comment added           |
| 11| Low       | Probabilities clipped before `log_loss`                              | Applied                 |
| 1 | Critical  | Point-in-time training features                                      | **Documented only** — refactor required |
| 2 | Critical  | Eliminate constants / collinear features in training                 | **Documented only**     |
| 9 | Medium    | Feature-list divergence detection at inference                       | **Documented only**     |

Items 1, 2, and 9 are the next priorities and require coordinated work
across `data.py`, `features.py`, and the season cache files in
`models/cache/features_*.parquet`. After those land, every metric in
`docs/data/stats.json` will need to be re-baselined because today's
numbers reflect look-ahead bias.
