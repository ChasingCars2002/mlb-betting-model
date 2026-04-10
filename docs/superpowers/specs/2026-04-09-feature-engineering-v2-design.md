# Feature Engineering V2 — Design Spec

**Date:** 2026-04-09  
**Status:** Approved

---

## Problem

The current feature engineering pipeline has a fundamental data leakage issue: every training game uses season-level aggregate stats (xFIP, SIERA, wRC+, OPS) that were computed at season end, not at the time the game was played. A game on April 5th receives the same xFIP as a game on September 20th. This means the model learns from information that didn't exist at prediction time, producing optimistic training metrics that don't hold in production.

Additionally, the model is missing meaningful signal: park run environment, pitcher rest and fatigue, true recent team offensive form, and actual lineup quality.

---

## Goals

- Eliminate data leakage by computing all training features from only information available before each game
- Add park factors, rest/fatigue, rolling team offense, and lineup quality features
- Keep training and prediction using identical feature computation logic (no train/serve skew)
- Validate improvement with a temporal holdout comparison (old vs. new Brier score / log loss)

---

## Approach

Replace season-level FanGraphs stat lookups with MLB Stats API game logs. For training, load all pitcher and team batting game logs for each season upfront, then process games in date order — computing rolling stats from only prior games. For prediction, fetch the same rolling stats live from the API.

Park factors are a hardcoded lookup. Rest/fatigue is derived from schedule data already in the pipeline.

---

## New Feature Columns (35 total, replacing 24)

### Starting Pitchers — 20 features

Two rolling windows per pitcher (home and away):

```
{side}_p_ERA_3,  {side}_p_WHIP_3,  {side}_p_K_pct_3,  {side}_p_BB_pct_3,  {side}_p_K_BB_pct_3
{side}_p_ERA_10, {side}_p_WHIP_10, {side}_p_K_pct_10, {side}_p_BB_pct_10, {side}_p_K_BB_pct_10
```

- `_3` = last 3 starts (hot/cold signal)
- `_10` = last 10 starts (sustainable form)
- K% = K / BF, BB% = BB / BF, K-BB% = K% − BB%

### Bullpen — 4 features

```
home_bullpen_era_10, home_bullpen_whip_10
away_bullpen_era_10, away_bullpen_whip_10
```

Rolling 10 relief appearances per team before game date.

### Team Offense — 4 features

```
home_team_ops_10, home_team_obp_10
away_team_ops_10, away_team_obp_10
```

Rolling last 10 games of team batting from MLB Stats API game logs.

### Lineup Quality — 2 features

```
home_lineup_ops, away_lineup_ops
```

Training: actual lineup OPS computed from boxscore (players who started that game).  
Prediction: projected lineup OPS from MLB Stats API pre-game data; falls back to `team_ops_10` if unavailable.

### Park Factor — 1 feature

```
park_factor
```

Home team's ballpark run factor. 1.0 = neutral park. Hardcoded lookup, updated annually.

### Rest / Fatigue — 4 features

```
home_pitcher_rest_days   (days since home starter's last start)
away_pitcher_rest_days
home_team_back_to_back   (1 if home team played yesterday, else 0)
away_team_back_to_back
```

---

## Park Factors Lookup

Hardcoded dict in `features.py`, updated manually once per season:

```python
PARK_FACTORS = {
    "COL": 1.24,
    "CIN": 1.10, "TEX": 1.09, "BOS": 1.08, "PHI": 1.07,
    "MIL": 1.05, "CHC": 1.04, "ATL": 1.03, "HOU": 1.02,
    "BAL": 1.02, "NYY": 1.01, "TOR": 1.01, "LAD": 1.00,
    "STL": 1.00, "WSH": 0.99, "MIN": 0.99, "DET": 0.99,
    "CLE": 0.98, "AZ":  0.98, "NYM": 0.98, "CHW": 0.97,
    "KC":  0.97, "LAA": 0.97, "SF":  0.96, "SEA": 0.96,
    "PIT": 0.96, "TB":  0.95, "MIA": 0.95, "SD":  0.94,
    "ATH": 0.93, "OAK": 0.93,
}
```

Missing team → defaults to 1.0 (neutral).

---

## Data Pipeline

### New Cache Files

```
models/cache/pitcher_logs_{season}.parquet
  columns: pitcher_id, pitcher_name, team_id, game_date, game_id,
           IP, H, ER, BB, K, HR, BF, is_start

models/cache/team_batting_logs_{season}.parquet
  columns: team_id, game_date, game_id, R, H, BB, K, PA, OBP, SLG, OPS

models/cache/lineups_{season}.parquet
  columns: game_id, team_id, game_date, lineup_ops
```

Completed seasons (year < current year) are cached permanently. Current season is always re-fetched on `--retrain`.

### New `data.py` Functions

**`get_pitcher_logs_season(season: int) -> pd.DataFrame`**  
Fetches per-start and relief appearance stats for all pitchers in a season via MLB Stats API schedule + boxscores. Returns pitcher_logs DataFrame. Called once per season; result cached to parquet.

**`get_team_batting_logs_season(season: int) -> pd.DataFrame`**  
Fetches per-game team batting stats for all teams via MLB Stats API. Returns team_batting_logs DataFrame. Cached per season.

**`get_lineup_ops_season(season: int, historical_games: pd.DataFrame) -> pd.DataFrame`**  
For each completed game, fetches the boxscore lineups and computes lineup OPS using each batter's season stats at time of game. Returns lineups DataFrame. Cached per season.

**`get_pitcher_recent_logs(pitcher_id: int, n: int = 10) -> pd.DataFrame`**  
Live fetch of a pitcher's last `n` appearances from MLB Stats API. Used by prediction path only.

**`get_team_recent_batting(team_id: int, n: int = 10) -> pd.DataFrame`**  
Live fetch of a team's last `n` games' batting stats. Used by prediction path only.

### Removed `data.py` Functions

`get_pitcher_stats`, `get_bullpen_stats`, `get_team_hitting_splits` — removed. No longer needed once both training and prediction paths use game logs.

---

## Feature Computation Functions (`features.py`)

**`compute_pitcher_rolling(logs: pd.DataFrame, as_of_date: str) -> dict`**  
Filters `logs` to `is_start=True` rows before `as_of_date`. Computes ERA/WHIP/K%/BB%/K-BB% for last 3 and last 10 starts. Returns dict of 10 values.

**`compute_bullpen_rolling(logs: pd.DataFrame, team_id: int, as_of_date: str) -> dict`**  
Filters `logs` to `is_start=False, team_id=team_id` rows before `as_of_date`. Computes ERA/WHIP from last 10 relief appearances. Returns dict of 2 values.

**`compute_team_offense_rolling(logs: pd.DataFrame, team_id: int, as_of_date: str) -> dict`**  
Filters batting logs to `team_id` rows before `as_of_date`. Computes OPS/OBP from last 10 games. Returns dict of 2 values.

**`compute_rest(pitcher_logs: pd.DataFrame, pitcher_id: int, team_batting_logs: pd.DataFrame, team_id: int, game_date: str) -> dict`**  
Returns `pitcher_rest_days` (days since last start, default 4 if unknown) and `team_back_to_back` (1 if team played on game_date - 1 day).

---

## Training Pipeline

`build_training_features(historical_games, pitcher_logs, team_batting_logs, lineups)` replaces the current implementation:

```
1. Sort historical_games by game_date ascending
2. For each game:
   a. compute_pitcher_rolling(pitcher_logs filtered to home pitcher_id, game_date)
   b. compute_pitcher_rolling(pitcher_logs filtered to away pitcher_id, game_date)
   c. compute_bullpen_rolling(pitcher_logs, home_team_id, game_date)
   d. compute_bullpen_rolling(pitcher_logs, away_team_id, game_date)
   e. compute_team_offense_rolling(team_batting_logs, home_team_id, game_date)
   f. compute_team_offense_rolling(team_batting_logs, away_team_id, game_date)
   g. park_factor = PARK_FACTORS.get(home_team, 1.0)
   h. compute_rest(pitcher_logs, home_pitcher_id, team_batting_logs, home_team_id, game_date)
   i. compute_rest(pitcher_logs, away_pitcher_id, team_batting_logs, away_team_id, game_date)
   j. lineup_ops from lineups parquet join on game_id + team_id
3. Assemble feature row, append to list
4. Return pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS), y
```

The function signature changes to accept pre-loaded DataFrames (injected by `get_or_build_season_features`), keeping I/O out of feature computation.

---

## Prediction Pipeline

`build_game_features(game)` is updated to use the same computation functions:

```
1. get_pitcher_recent_logs(home_pitcher_id, n=10) → home_p_logs
2. compute_pitcher_rolling(home_p_logs, today)
3. Same for away pitcher
4. get_team_recent_batting(home_team_id, n=10) → home_batting
5. compute_team_offense_rolling(home_batting, ..., today)
6. Same for away team
7. compute_bullpen_rolling(live bullpen logs, ...)
8. park_factor lookup
9. compute_rest(...)
10. lineup_ops from pre-game projected lineup (fall back to team OPS if unavailable)
```

---

## Error Handling & Defaults

| Condition | Default |
|---|---|
| Pitcher has 0 starts in window | ERA 4.20, WHIP 1.30, K% 21%, BB% 8%, K-BB% 13% |
| Pitcher has < 3 starts (short window) | Use all available starts |
| Pitcher has < 10 starts (long window) | Use all available starts |
| Team has < 3 batting games | OPS 0.720, OBP 0.315 |
| Lineup data unavailable | team_ops_10; if also unavailable, OPS 0.720 |
| Park factor missing | 1.0 (neutral) |
| Pitcher rest unknown | 4 days (median) |

All defaults logged at DEBUG level.

---

## Validation Step

After implementation, run a temporal holdout comparison before declaring the new model production-ready:

1. Train old model on 2023–2024 using existing feature pipeline
2. Train new model on 2023–2024 using new feature pipeline
3. Evaluate both on 2025 (held out, never seen during training)
4. Compare: Brier score, log loss, accuracy
5. New model must match or beat old model on all three metrics to replace it

Run this comparison once manually after implementation, before declaring the new model production-ready. It is not a recurring CLI flag.

---

## Files Changed

| File | Action |
|---|---|
| `data.py` | Add 5 new functions; remove 3 old ones |
| `features.py` | Replace FEATURE_COLUMNS, add 4 computation functions, add PARK_FACTORS, rewrite `build_training_features`, update `build_game_features` |
| `tests/test_features.py` | New — unit tests for all computation functions and defaults |

**Unchanged:** `train.py`, `model.py`, `main.py`, `database.py`, `evaluate.py`, `odds.py`, `config.py`

---

## Out of Scope

- Umpire tendencies
- Weather data
- Statcast-derived metrics (xFIP, SIERA, spin rate)
- Kelly criterion / bet sizing
- Profitability backtesting
- Hyperparameter tuning / model architecture changes
