# Incremental Retrain Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--retrain` CLI command and weekly scheduler job that retrains models using cached per-season feature matrices, skipping completed seasons (2023–2025) on every run.

**Architecture:** Per-season `(X, y)` feature DataFrames are saved to `models/cache/features_{year}.parquet` after the first build. On `--retrain`, completed seasons load from cache instantly while only the current season hits the MLB and FanGraphs APIs. A `training_state.json` file tracks the last run, season row counts, and a hash of `FEATURE_COLUMNS` to detect schema changes that require a full cache invalidation.

**Tech Stack:** Python 3.11+, pandas (parquet I/O), hashlib (feature hash), json, APScheduler (existing), joblib (existing), pytest + unittest.mock

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `config.py` | Modify | Add `CACHE_DIR`, `TRAINING_STATE_PATH`, retrain schedule constants |
| `train.py` | Modify | Add helpers + `run_incremental_retrain()`; update `main()` |
| `main.py` | Modify | Add `--retrain`/`--force` args, `run_retrain()`, weekly scheduler job |
| `tests/test_train.py` | Create | Unit tests for all new `train.py` functions |

---

## Task 1: Add config constants

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Add the four new constants to `config.py` after the `MODEL_DIR` line**

Open `config.py`. After the line `MODEL_DIR = BASE_DIR / "models"`, add:

```python
CACHE_DIR = MODEL_DIR / "cache"
TRAINING_STATE_PATH = MODEL_DIR / "training_state.json"

# --- Retrain Scheduler ---
RETRAIN_SCHEDULE_DAY = "mon"
RETRAIN_SCHEDULE_HOUR = 6
RETRAIN_SCHEDULE_MINUTE = 0
```

- [ ] **Step 2: Verify the import works**

```bash
cd C:/Users/patri/Claude/mlb-betting-model
python -c "from config import CACHE_DIR, TRAINING_STATE_PATH, RETRAIN_SCHEDULE_DAY, RETRAIN_SCHEDULE_HOUR, RETRAIN_SCHEDULE_MINUTE; print('OK', CACHE_DIR)"
```

Expected output:
```
OK C:\Users\patri\Claude\mlb-betting-model\models\cache
```

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add cache dir and retrain schedule constants to config"
```

---

## Task 2: Training state helpers + feature hash

**Files:**
- Modify: `train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write failing tests for `get_feature_columns_hash`, `load_training_state`, and `save_training_state`**

Create `tests/test_train.py`:

```python
"""Tests for incremental retrain helpers in train.py."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from features import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_feature_df(n=20, seed=42):
    """Return a synthetic (X, y) pair with correct feature columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.uniform(3.0, 6.0, size=(n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    y = pd.Series(rng.integers(0, 2, n), name="home_win")
    return X, y


# ---------------------------------------------------------------------------
# get_feature_columns_hash
# ---------------------------------------------------------------------------

class TestGetFeatureColumnsHash:
    def test_returns_string(self):
        from train import get_feature_columns_hash
        assert isinstance(get_feature_columns_hash(), str)

    def test_deterministic(self):
        from train import get_feature_columns_hash
        assert get_feature_columns_hash() == get_feature_columns_hash()

    def test_changes_when_columns_change(self):
        import train as train_mod
        original = train_mod.get_feature_columns_hash()
        with patch.object(train_mod, "FEATURE_COLUMNS", FEATURE_COLUMNS + ["extra_col"]):
            changed = train_mod.get_feature_columns_hash()
        assert original != changed


# ---------------------------------------------------------------------------
# load_training_state / save_training_state
# ---------------------------------------------------------------------------

class TestTrainingState:
    def test_load_returns_empty_when_file_missing(self, tmp_path):
        import train as train_mod
        with patch.object(train_mod, "TRAINING_STATE_PATH", tmp_path / "nonexistent.json"):
            assert train_mod.load_training_state() == {}

    def test_load_returns_empty_on_malformed_json(self, tmp_path):
        import train as train_mod
        bad = tmp_path / "state.json"
        bad.write_text("not json {{{")
        with patch.object(train_mod, "TRAINING_STATE_PATH", bad):
            assert train_mod.load_training_state() == {}

    def test_save_creates_file(self, tmp_path):
        import train as train_mod
        state_path = tmp_path / "state.json"
        with patch.object(train_mod, "TRAINING_STATE_PATH", state_path):
            train_mod.save_training_state({"key": "value"})
        assert state_path.exists()

    def test_roundtrip(self, tmp_path):
        import train as train_mod
        state_path = tmp_path / "state.json"
        state = {"last_trained": "2026-04-07T06:00:00", "feature_columns_hash": "abc123"}
        with patch.object(train_mod, "TRAINING_STATE_PATH", state_path):
            train_mod.save_training_state(state)
            loaded = train_mod.load_training_state()
        assert loaded == state
```

- [ ] **Step 2: Run tests to confirm they fail with ImportError**

```bash
cd C:/Users/patri/Claude/mlb-betting-model
python -m pytest tests/test_train.py::TestGetFeatureColumnsHash tests/test_train.py::TestTrainingState -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'get_feature_columns_hash' from 'train'`

- [ ] **Step 3: Add the three functions to `train.py`**

At the top of `train.py`, replace the existing imports block with:

```python
"""One-time training script — pull historical data, engineer features, train models."""

import hashlib
import json
import logging
import sys

import pandas as pd
from datetime import date, datetime

from config import TRAINING_SEASONS, LOG_FILE, CACHE_DIR, TRAINING_STATE_PATH
from data import get_historical_game_data
from features import build_training_features, FEATURE_COLUMNS
from model import train_models
```

Then, after the `logger = logging.getLogger(__name__)` line, add the three helpers:

```python
def get_feature_columns_hash() -> str:
    """Return a 16-char SHA256 of FEATURE_COLUMNS for cache invalidation."""
    content = ",".join(FEATURE_COLUMNS)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_training_state() -> dict:
    """Read training_state.json. Returns {} if missing or malformed."""
    if not TRAINING_STATE_PATH.exists():
        return {}
    try:
        with open(TRAINING_STATE_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read training_state.json: %s. Treating as first run.", e)
        return {}


def save_training_state(state: dict):
    """Write state dict to training_state.json."""
    TRAINING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    logger.info("Training state saved to %s", TRAINING_STATE_PATH)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_train.py::TestGetFeatureColumnsHash tests/test_train.py::TestTrainingState -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: add feature hash and training state helpers to train.py"
```

---

## Task 3: Per-season feature cache (`get_or_build_season_features`)

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Add tests for `get_or_build_season_features` to `tests/test_train.py`**

Append to `tests/test_train.py`:

```python
# ---------------------------------------------------------------------------
# get_or_build_season_features
# ---------------------------------------------------------------------------

class TestGetOrBuildSeasonFeatures:
    def test_builds_and_writes_cache_on_first_call(self, tmp_path):
        import train as train_mod
        X, y = make_feature_df(20)
        cache_dir = tmp_path / "cache"

        with patch.object(train_mod, "CACHE_DIR", cache_dir), \
             patch("train.get_historical_game_data", return_value=pd.DataFrame({"dummy": [1]})), \
             patch("train.build_training_features", return_value=(X, y)):
            result_X, result_y = train_mod.get_or_build_season_features(
                2023, force_rebuild=False, current_hash="abc"
            )

        assert (cache_dir / "features_2023.parquet").exists()
        pd.testing.assert_frame_equal(result_X.reset_index(drop=True), X.reset_index(drop=True))
        assert len(result_y) == 20

    def test_loads_from_cache_without_api_calls(self, tmp_path):
        import train as train_mod
        X, y = make_feature_df(20)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        df = X.copy()
        df["home_win"] = y.values
        df.to_parquet(cache_dir / "features_2023.parquet", index=False)

        with patch.object(train_mod, "CACHE_DIR", cache_dir), \
             patch("train.get_historical_game_data") as mock_fetch:
            result_X, result_y = train_mod.get_or_build_season_features(
                2023, force_rebuild=False, current_hash="abc"
            )
            mock_fetch.assert_not_called()

        assert len(result_X) == 20

    def test_force_rebuild_ignores_existing_cache(self, tmp_path):
        import train as train_mod
        X_stale, y_stale = make_feature_df(20, seed=1)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        df = X_stale.copy()
        df["home_win"] = y_stale.values
        df.to_parquet(cache_dir / "features_2023.parquet", index=False)

        X_fresh, y_fresh = make_feature_df(30, seed=99)
        with patch.object(train_mod, "CACHE_DIR", cache_dir), \
             patch("train.get_historical_game_data", return_value=pd.DataFrame({"dummy": [1]})), \
             patch("train.build_training_features", return_value=(X_fresh, y_fresh)):
            result_X, _ = train_mod.get_or_build_season_features(
                2023, force_rebuild=True, current_hash="abc"
            )

        assert len(result_X) == 30

    def test_corrupt_cache_falls_back_to_rebuild(self, tmp_path):
        import train as train_mod
        X, y = make_feature_df(20)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "features_2023.parquet").write_bytes(b"not a parquet file")

        with patch.object(train_mod, "CACHE_DIR", cache_dir), \
             patch("train.get_historical_game_data", return_value=pd.DataFrame({"dummy": [1]})), \
             patch("train.build_training_features", return_value=(X, y)):
            result_X, _ = train_mod.get_or_build_season_features(
                2023, force_rebuild=False, current_hash="abc"
            )

        assert len(result_X) == 20

    def test_returns_empty_when_no_game_data(self, tmp_path):
        import train as train_mod
        cache_dir = tmp_path / "cache"

        with patch.object(train_mod, "CACHE_DIR", cache_dir), \
             patch("train.get_historical_game_data", return_value=pd.DataFrame()):
            result_X, result_y = train_mod.get_or_build_season_features(
                2026, force_rebuild=True, current_hash="abc"
            )

        assert result_X.empty
        assert len(result_y) == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_train.py::TestGetOrBuildSeasonFeatures -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'get_or_build_season_features' from 'train'`

- [ ] **Step 3: Add `get_or_build_season_features` to `train.py`**

After `save_training_state`, add:

```python
def get_or_build_season_features(
    season: int,
    force_rebuild: bool,
    current_hash: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load season features from cache, or build from scratch if unavailable.

    Returns (X, y). Returns empty DataFrame/Series when no game data exists
    (e.g., off-season or season not yet started).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"features_{season}.parquet"

    if not force_rebuild and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            X = df[FEATURE_COLUMNS]
            y = df["home_win"].rename("home_win")
            logger.info("Loaded cached features for %d (%d rows).", season, len(X))
            return X, y
        except Exception as e:
            logger.warning("Cache for %d is corrupt (%s) — rebuilding.", season, e)

    logger.info("Building features for %d season...", season)
    historical = get_historical_game_data([season])

    if historical.empty:
        logger.warning("No game data found for %d — skipping.", season)
        return (
            pd.DataFrame(columns=FEATURE_COLUMNS),
            pd.Series([], dtype=int, name="home_win"),
        )

    X, y = build_training_features(historical)

    cache_df = X.copy()
    cache_df["home_win"] = y.values
    cache_df.to_parquet(cache_path, index=False)
    logger.info("Cached features for %d to %s.", season, cache_path)

    return X, y
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_train.py::TestGetOrBuildSeasonFeatures -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: add per-season feature cache with get_or_build_season_features"
```

---

## Task 4: `run_incremental_retrain()` + tests

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Add tests for `run_incremental_retrain` to `tests/test_train.py`**

Append to `tests/test_train.py`:

```python
# ---------------------------------------------------------------------------
# run_incremental_retrain
# ---------------------------------------------------------------------------

class TestRunIncrementalRetrain:
    def _patch_context(self, tmp_path, seasons=None):
        """Return a dict of patches for run_incremental_retrain tests."""
        import train as train_mod
        return {
            "TRAINING_STATE_PATH": tmp_path / "state.json",
            "CACHE_DIR": tmp_path / "cache",
            "TRAINING_SEASONS": seasons or [2023, 2024],
        }

    def test_combines_multiple_seasons_and_trains(self, tmp_path):
        import train as train_mod
        X, y = make_feature_df(30)
        ctx = self._patch_context(tmp_path, seasons=[2023, 2024])

        call_count = {"n": 0}
        def fake_build(season, force_rebuild, current_hash):
            call_count["n"] += 1
            return X, y

        with patch.object(train_mod, "TRAINING_STATE_PATH", ctx["TRAINING_STATE_PATH"]), \
             patch.object(train_mod, "CACHE_DIR", ctx["CACHE_DIR"]), \
             patch.object(train_mod, "TRAINING_SEASONS", ctx["TRAINING_SEASONS"]), \
             patch("train.get_or_build_season_features", side_effect=fake_build), \
             patch("train.train_models") as mock_train:
            train_mod.run_incremental_retrain(force=False, current_year=2026)

        # 2023, 2024 (base) + 2026 (current) = 3 calls
        assert call_count["n"] == 3
        mock_train.assert_called_once()
        combined_X = mock_train.call_args[0][0]
        assert len(combined_X) == 90  # 3 seasons × 30 rows

    def test_current_season_always_force_rebuilt(self, tmp_path):
        import train as train_mod
        X, y = make_feature_df(30)
        ctx = self._patch_context(tmp_path, seasons=[2023])
        rebuild_by_season = {}

        def fake_build(season, force_rebuild, current_hash):
            rebuild_by_season[season] = force_rebuild
            return X, y

        with patch.object(train_mod, "TRAINING_STATE_PATH", ctx["TRAINING_STATE_PATH"]), \
             patch.object(train_mod, "CACHE_DIR", ctx["CACHE_DIR"]), \
             patch.object(train_mod, "TRAINING_SEASONS", ctx["TRAINING_SEASONS"]), \
             patch("train.get_or_build_season_features", side_effect=fake_build), \
             patch("train.train_models"):
            train_mod.run_incremental_retrain(force=False, current_year=2026)

        assert rebuild_by_season[2023] is False   # completed season uses cache
        assert rebuild_by_season[2026] is True    # current season always rebuilds

    def test_force_rebuilds_all_seasons(self, tmp_path):
        import train as train_mod
        X, y = make_feature_df(30)
        ctx = self._patch_context(tmp_path, seasons=[2023, 2024])
        rebuild_flags = []

        def fake_build(season, force_rebuild, current_hash):
            rebuild_flags.append(force_rebuild)
            return X, y

        with patch.object(train_mod, "TRAINING_STATE_PATH", ctx["TRAINING_STATE_PATH"]), \
             patch.object(train_mod, "CACHE_DIR", ctx["CACHE_DIR"]), \
             patch.object(train_mod, "TRAINING_SEASONS", ctx["TRAINING_SEASONS"]), \
             patch("train.get_or_build_season_features", side_effect=fake_build), \
             patch("train.train_models"):
            train_mod.run_incremental_retrain(force=True, current_year=2026)

        assert all(rebuild_flags), "force=True should rebuild every season"

    def test_hash_mismatch_triggers_force_rebuild(self, tmp_path):
        import train as train_mod
        X, y = make_feature_df(30)
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({"feature_columns_hash": "stale_hash_000"}))
        cache_dir = tmp_path / "cache"
        rebuild_flags = []

        def fake_build(season, force_rebuild, current_hash):
            rebuild_flags.append(force_rebuild)
            return X, y

        with patch.object(train_mod, "TRAINING_STATE_PATH", state_path), \
             patch.object(train_mod, "CACHE_DIR", cache_dir), \
             patch.object(train_mod, "TRAINING_SEASONS", [2023]), \
             patch("train.get_or_build_season_features", side_effect=fake_build), \
             patch("train.train_models"):
            train_mod.run_incremental_retrain(force=False, current_year=2026)

        assert all(rebuild_flags), "Hash mismatch must force a full rebuild"

    def test_skips_empty_season_and_still_trains(self, tmp_path):
        import train as train_mod
        X, y = make_feature_df(30)
        ctx = self._patch_context(tmp_path, seasons=[2023])

        def fake_build(season, force_rebuild, current_hash):
            if season == 2026:
                return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series([], dtype=int)
            return X, y

        with patch.object(train_mod, "TRAINING_STATE_PATH", ctx["TRAINING_STATE_PATH"]), \
             patch.object(train_mod, "CACHE_DIR", ctx["CACHE_DIR"]), \
             patch.object(train_mod, "TRAINING_SEASONS", ctx["TRAINING_SEASONS"]), \
             patch("train.get_or_build_season_features", side_effect=fake_build), \
             patch("train.train_models") as mock_train:
            train_mod.run_incremental_retrain(force=False, current_year=2026)

        combined_X = mock_train.call_args[0][0]
        assert len(combined_X) == 30  # only 2023, 2026 was empty

    def test_saves_training_state_after_run(self, tmp_path):
        import train as train_mod
        X, y = make_feature_df(30)
        ctx = self._patch_context(tmp_path, seasons=[2023])

        with patch.object(train_mod, "TRAINING_STATE_PATH", ctx["TRAINING_STATE_PATH"]), \
             patch.object(train_mod, "CACHE_DIR", ctx["CACHE_DIR"]), \
             patch.object(train_mod, "TRAINING_SEASONS", ctx["TRAINING_SEASONS"]), \
             patch("train.get_or_build_season_features", return_value=(X, y)), \
             patch("train.train_models"):
            train_mod.run_incremental_retrain(force=False, current_year=2026)

        assert ctx["TRAINING_STATE_PATH"].exists()
        state = json.loads(ctx["TRAINING_STATE_PATH"].read_text())
        assert "last_trained" in state
        assert "feature_columns_hash" in state
        assert "2023" in state["seasons"]
        assert "2026" in state["seasons"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_train.py::TestRunIncrementalRetrain -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'run_incremental_retrain' from 'train'`

- [ ] **Step 3: Add `run_incremental_retrain` to `train.py`**

After `get_or_build_season_features`, add:

```python
def run_incremental_retrain(force: bool = False, current_year: int = None):
    """Retrain models using cached features for completed seasons.

    Completed seasons (year < current_year) load from parquet cache.
    The current season always re-fetches and rebuilds features.
    Passing force=True or detecting a feature schema change triggers
    a full rebuild of all season caches.

    Args:
        force: Wipe all caches and rebuild from scratch.
        current_year: Override the current year (used in tests).
    """
    if current_year is None:
        current_year = date.today().year

    print(f"\n{'='*60}")
    label = "Full Rebuild" if force else "Incremental Retrain"
    print(f"  MLB BETTING MODEL — {label}")
    print(f"{'='*60}")

    current_hash = get_feature_columns_hash()
    state = load_training_state()

    stored_hash = state.get("feature_columns_hash", "")
    if stored_hash and stored_hash != current_hash:
        print("\n  WARNING: Feature schema changed — rebuilding all season caches.")
        logger.warning(
            "Feature hash mismatch: stored=%s current=%s. Forcing full rebuild.",
            stored_hash, current_hash,
        )
        force = True

    seasons = list(TRAINING_SEASONS)
    if current_year not in seasons:
        seasons.append(current_year)

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []
    season_stats: dict[str, dict] = {}

    for season in seasons:
        is_current = (season == current_year)
        rebuild = force or is_current
        label = "rebuilding" if rebuild else "from cache"
        print(f"\n  Season {season}: {label}...")

        X, y = get_or_build_season_features(
            season, force_rebuild=rebuild, current_hash=current_hash
        )

        if X.empty:
            print(f"  Season {season}: no data available, skipping.")
            logger.info("No data for season %d, skipping.", season)
            continue

        all_X.append(X)
        all_y.append(y)
        season_stats[str(season)] = {"rows": len(X), "cached": not rebuild}
        print(f"           {len(X)} games loaded.")

    if not all_X:
        print("\n  ERROR: No training data available across all seasons. Aborting.")
        logger.error("No training data available. Aborting retrain.")
        return

    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)

    print(f"\n  Combined: {len(X_combined)} games across {len(all_X)} seasons.")
    print(f"  Home win rate: {y_combined.mean():.3f}\n")

    train_models(X_combined, y_combined)

    new_state = {
        "last_trained": datetime.now().isoformat(timespec="seconds"),
        "feature_columns_hash": current_hash,
        "seasons": season_stats,
    }
    save_training_state(new_state)
    print("\n  Training state saved. Models ready.")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_train.py::TestRunIncrementalRetrain -v
```

Expected: `6 passed`

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
python -m pytest tests/ -v
```

Expected: all existing tests still pass, plus the 6 new ones.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: add run_incremental_retrain with per-season feature caching"
```

---

## Task 5: Wire up `train.main()`, CLI flags, and weekly scheduler

**Files:**
- Modify: `train.py` (update `main()`)
- Modify: `main.py` (add `--retrain`/`--force`, `run_retrain()`, weekly scheduler job)

- [ ] **Step 1: Update `train.py::main()` to call `run_incremental_retrain(force=True)`**

In `train.py`, replace the existing `main()` function:

```python
def main():
    print(f"\nMLB Betting Model — Full Training Pipeline")
    print(f"Seasons: {TRAINING_SEASONS} + current year (auto-detected)")
    print("=" * 50)
    run_incremental_retrain(force=True)
```

This ensures `--train` writes per-season caches after a full rebuild, so the next
`--retrain` can immediately use them.

- [ ] **Step 2: Verify `train.py::main()` still runs cleanly (import check)**

```bash
python -c "from train import main, run_incremental_retrain; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Add `--retrain` and `--force` args to `main.py`**

In `main.py`, open the `argparse` block. After the existing `--train` argument, add:

```python
    parser.add_argument(
        "--retrain", action="store_true",
        help="Incremental retrain: use cached features for completed seasons, rebuild current season only.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="With --retrain: wipe all caches and do a full rebuild from scratch.",
    )
```

- [ ] **Step 4: Add `run_retrain()` function to `main.py`**

In `main.py`, after the `show_stats()` function and before `run_scheduler()`, add:

```python
def run_retrain(force: bool = False):
    """Incremental model retrain entry point (used by CLI and scheduler)."""
    from train import run_incremental_retrain
    run_incremental_retrain(force=force)
```

- [ ] **Step 5: Wire `--retrain` into the `main()` dispatch block in `main.py`**

In `main.py::main()`, in the `if/elif` chain that dispatches on `args`, add the `--retrain` branch after `--train`:

```python
    if args.train:
        from train import main as train_main
        train_main()
    elif args.retrain:
        run_retrain(force=args.force)
    elif args.run_now:
```

Also update the help examples block at the bottom of `main()`:

```python
        print("\nExamples:")
        print("  python main.py --train               # Full retrain (wipes cache)")
        print("  python main.py --retrain             # Incremental retrain (uses cache)")
        print("  python main.py --retrain --force     # Force full rebuild via --retrain")
        print("  python main.py --run-now             # Run today's predictions")
        print("  python main.py --grade               # Grade yesterday's picks")
        print("  python main.py --stats               # Show lifetime stats")
        print("  python main.py --schedule            # Start daily scheduler")
```

- [ ] **Step 6: Add retrain schedule constants to the import block in `main.py`**

In `main.py`, update the config import line to include the new constants:

```python
from config import (
    LOG_FILE,
    MORNING_RUN_HOUR,
    MORNING_RUN_MINUTE,
    GRADING_HOUR,
    GRADING_MINUTE,
    RETRAIN_SCHEDULE_DAY,
    RETRAIN_SCHEDULE_HOUR,
    RETRAIN_SCHEDULE_MINUTE,
)
```

- [ ] **Step 7: Add the weekly retrain job to `run_scheduler()` in `main.py`**

In `main.py::run_scheduler()`, update the print block and add the new job. Replace the existing function with:

```python
def run_scheduler():
    """Run APScheduler to automate morning predictions, grading, and weekly retrain."""
    print("\n  MLB Betting Model — Scheduler Starting")
    print(f"  Predictions: Daily at {MORNING_RUN_HOUR:02d}:{MORNING_RUN_MINUTE:02d} ET")
    print(f"  Grading:     Daily at {GRADING_HOUR:02d}:{GRADING_MINUTE:02d} ET")
    print(f"  Retrain:     Weekly {RETRAIN_SCHEDULE_DAY.capitalize()} at {RETRAIN_SCHEDULE_HOUR:02d}:{RETRAIN_SCHEDULE_MINUTE:02d} ET")
    print("  Press Ctrl+C to stop.\n")

    scheduler = BlockingScheduler(timezone="US/Eastern")

    scheduler.add_job(
        run_predictions,
        CronTrigger(hour=MORNING_RUN_HOUR, minute=MORNING_RUN_MINUTE),
        id="daily_predictions",
        name="Daily MLB Predictions",
    )

    scheduler.add_job(
        run_grading,
        CronTrigger(hour=GRADING_HOUR, minute=GRADING_MINUTE),
        id="daily_grading",
        name="Daily Prediction Grading",
    )

    scheduler.add_job(
        run_retrain,
        CronTrigger(
            day_of_week=RETRAIN_SCHEDULE_DAY,
            hour=RETRAIN_SCHEDULE_HOUR,
            minute=RETRAIN_SCHEDULE_MINUTE,
        ),
        id="weekly_retrain",
        name="Weekly Incremental Retrain",
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n  Scheduler stopped.")
```

- [ ] **Step 8: Smoke-test the CLI**

```bash
python main.py --help
```

Expected output includes:
```
  --retrain             Incremental retrain: use cached features for completed
  --force               With --retrain: wipe all caches and do a full rebuild
```

- [ ] **Step 9: Run the full test suite one final time**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass, no regressions.

- [ ] **Step 10: Commit**

```bash
git add train.py main.py
git commit -m "feat: wire up --retrain CLI flag and weekly scheduler job"
```
