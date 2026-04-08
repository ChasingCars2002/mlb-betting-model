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
