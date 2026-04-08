# Incremental Retrain Workflow — Design Spec

**Date:** 2026-04-07  
**Status:** Approved

---

## Problem

`python main.py --train` re-downloads and re-processes all historical seasons (2023–2025)
every time it runs. This makes weekly retraining impractical — completed seasons never
change, yet their data is re-fetched and their features are re-engineered from scratch
on every run.

---

## Goals

- Weekly retraining that only processes new data (current season)
- Completed seasons (year < current year) processed once, cached permanently
- Auto-detect and include the current calendar year as an in-progress season
- Manual CLI trigger + optional weekly scheduler job
- Safe invalidation when feature schema changes

---

## Approach: Per-Season Feature Matrix Cache

After feature engineering, save each season's `(X, y)` as a parquet file in
`models/cache/`. On retrain, load completed seasons from cache — skipping all MLB API
and FanGraphs API calls for those seasons — and only rebuild features for the current
season's games.

---

## File Changes

### `config.py`
Add:
```python
CACHE_DIR = MODEL_DIR / "cache"
TRAINING_STATE_PATH = MODEL_DIR / "training_state.json"
RETRAIN_SCHEDULE_DAY = "mon"   # day-of-week for weekly scheduler job
RETRAIN_SCHEDULE_HOUR = 6      # hour (ET) for weekly retrain
RETRAIN_SCHEDULE_MINUTE = 0
```

`TRAINING_SEASONS` stays in `config.py` as the base historical seasons list.
Both `--train` and `--retrain` auto-append the current calendar year at runtime
so `config.py` never needs editing when a new season starts:
```python
TRAINING_SEASONS = [2023, 2024, 2025]  # current year auto-appended at runtime
```

### `train.py`
Add incremental training logic:

**`get_feature_columns_hash() -> str`**  
Returns a short SHA256 of `FEATURE_COLUMNS` — used to detect schema changes that
require cache invalidation.

**`load_training_state() -> dict`**  
Reads `training_state.json`. Returns empty dict if missing or malformed.

**`save_training_state(state: dict)`**  
Writes updated metadata to `training_state.json`.

**`get_or_build_season_features(season: int, force_rebuild: bool, current_hash: str) -> tuple[pd.DataFrame, pd.Series]`**  
- If `force_rebuild=False` and cache file exists and hash matches: load from
  `models/cache/features_{season}.parquet`, return `(X, y)`
- Otherwise: call `get_historical_game_data([season])` + `build_training_features()`,
  save result to cache, return `(X, y)`

**`run_incremental_retrain(force: bool = False)`**  
Main entry point for `--retrain`:
1. Compute `current_hash = get_feature_columns_hash()`
2. Load `training_state.json`; detect hash mismatch → set `force=True` with warning
3. Build season list: `TRAINING_SEASONS + [current_year]` (deduplicated)
4. For each completed season (year < current year): call `get_or_build_season_features()`
   with `force_rebuild=force`
5. For current year: always call with `force_rebuild=True` (always re-fetch in-progress data)
6. If current year has zero games (off-season): skip, log notice
7. Concatenate all `(X, y)` DataFrames → call `train_models(X, y)`
8. Save updated `training_state.json`

### `main.py`
**CLI additions:**
```
python main.py --retrain           # incremental retrain (uses cache for completed seasons)
python main.py --retrain --force   # wipes cache, full rebuild
python main.py --train             # unchanged — full rebuild from scratch
```

`--train` calls existing `train_main()`, which also auto-appends current year to
`TRAINING_SEASONS` before calling `get_historical_game_data()`. Cache is written for
completed seasons after a full rebuild so the next `--retrain` can use it.

**Scheduler addition:**
```python
scheduler.add_job(
    run_incremental_retrain,
    CronTrigger(day_of_week=RETRAIN_SCHEDULE_DAY,
                hour=RETRAIN_SCHEDULE_HOUR,
                minute=RETRAIN_SCHEDULE_MINUTE),
    id="weekly_retrain",
    name="Weekly Incremental Retrain",
)
```

---

## Cache Layout

```
models/
  cache/
    features_2023.parquet   # columns: FEATURE_COLUMNS + "home_win"
    features_2024.parquet
    features_2025.parquet
    features_2026.parquet   # rebuilt every retrain
  training_state.json
  xgboost.joblib
  logistic_regression.joblib
  feature_medians.joblib
```

Each parquet has one row per game. Columns are all `FEATURE_COLUMNS` plus `home_win`
(the target), so a single file stores both X and y.

---

## `training_state.json` Schema

```json
{
  "last_trained": "2026-04-07T06:00:00",
  "feature_columns_hash": "a3f9c2...",
  "seasons": {
    "2023": {"rows": 2430, "cached": true},
    "2024": {"rows": 2430, "cached": true},
    "2025": {"rows": 2430, "cached": true},
    "2026": {"rows": 127,  "cached": false}
  }
}
```

---

## Cache Invalidation

| Trigger | Behavior |
|---|---|
| `feature_columns_hash` mismatch | Warn: `"Feature schema changed — rebuilding all season caches"`. Set `force=True`. |
| `--retrain --force` | Delete all cache parquets, full rebuild. |
| `--train` | Full rebuild (existing behavior), cache written for completed seasons afterward. |
| Corrupt/unreadable parquet | Log WARNING, rebuild that season, continue. |
| Off-season (0 current-year games) | Skip current year, train on completed seasons only, log notice. |

---

## Error Handling

- Missing or malformed `training_state.json`: treat as first run, full rebuild
- Corrupt parquet: fall back to full rebuild for that season, log WARNING
- Zero current-season games: skip cleanly, log INFO notice
- Feature hash mismatch: full rebuild with clear console message
- All cache misses/fallbacks logged at WARNING; normal cache hits at DEBUG

---

## Scheduler Summary

| Job | Schedule | Function |
|---|---|---|
| Daily predictions | 09:00 ET daily | `run_predictions()` |
| Daily grading | 08:00 ET daily | `run_grading()` |
| Weekly retrain | 06:00 ET Monday | `run_incremental_retrain()` |

---

## Out of Scope

- Caching raw game records (schedule/results) separately from features
- Online learning / partial model updates (full retrain on combined data each time)
- Model versioning or rollback
- Feature drift detection
