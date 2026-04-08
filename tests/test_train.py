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
