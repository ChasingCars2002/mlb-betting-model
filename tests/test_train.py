"""Tests for incremental retrain helpers in train.py."""

import json
import sys
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
                2023, force_rebuild=False
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
                2023, force_rebuild=False
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
                2023, force_rebuild=True
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
                2023, force_rebuild=False
            )

        assert len(result_X) == 20

    def test_returns_empty_when_no_game_data(self, tmp_path):
        import train as train_mod
        cache_dir = tmp_path / "cache"

        with patch.object(train_mod, "CACHE_DIR", cache_dir), \
             patch("train.get_historical_game_data", return_value=pd.DataFrame()):
            result_X, result_y = train_mod.get_or_build_season_features(
                2026, force_rebuild=True
            )

        assert result_X.empty
        assert len(result_y) == 0


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
        def fake_build(season, force_rebuild):
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

        def fake_build(season, force_rebuild):
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

        def fake_build(season, force_rebuild):
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

        def fake_build(season, force_rebuild):
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

        def fake_build(season, force_rebuild):
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


# ---------------------------------------------------------------------------
# Feature data-quality guard (regression)
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402
from features import (  # noqa: E402
    FALLBACK_VALUES, MAX_FALLBACK_RATE, check_feature_quality,
    feature_fallback_rates,
)


def _matrix(n=100, fallback_rows=0):
    """n rows of realistic values, with `fallback_rows` degraded to defaults."""
    rows = []
    for i in range(n):
        row = {}
        for col in FEATURE_COLUMNS:
            fb = FALLBACK_VALUES.get(col)
            if fb is None:                      # park_factor
                row[col] = 0.97 + (i % 5) * 0.01
            elif i < fallback_rows:
                row[col] = fb                   # upstream fetch failed
            else:
                row[col] = fb * (1.0 + ((i % 7) + 1) * 0.02)
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


class TestFeatureQualityGuard:
    """data.py swallows every upstream failure and returns a league-average
    constant. In a bulk training build that silently turns a broken endpoint
    into a full-size, signal-free matrix — which is exactly what the persisted
    medians show happened (every one equals its fallback constant).
    """

    def test_clean_matrix_has_no_fallbacks(self):
        rates = feature_fallback_rates(_matrix(fallback_rows=0))
        assert rates
        assert max(rates.values()) == 0.0

    def test_fully_degraded_matrix_is_detected(self):
        rates = feature_fallback_rates(_matrix(n=100, fallback_rows=100))
        assert min(rates.values()) == 1.0

    def test_rate_is_measured_per_feature(self):
        rates = feature_fallback_rates(_matrix(n=100, fallback_rows=40))
        for col, rate in rates.items():
            assert rate == pytest.approx(0.40, abs=1e-9), col

    def test_check_flags_only_above_threshold(self, caplog):
        import logging
        with caplog.at_level(logging.ERROR):
            check_feature_quality(_matrix(n=100, fallback_rows=10))
        assert "DATA QUALITY" not in caplog.text

        caplog.clear()
        with caplog.at_level(logging.ERROR):
            check_feature_quality(_matrix(n=100, fallback_rows=90))
        assert "DATA QUALITY" in caplog.text

    def test_park_factor_is_not_a_fallback_feature(self):
        # park_factor comes from a static table, never from a failing fetch.
        assert "park_factor" not in FALLBACK_VALUES

    def test_every_fetched_feature_has_a_known_fallback(self):
        fetched = [c for c in FEATURE_COLUMNS if c != "park_factor"]
        assert set(fetched) == set(FALLBACK_VALUES)

    def test_threshold_is_a_sane_fraction(self):
        assert 0.0 < MAX_FALLBACK_RATE < 1.0

    def test_empty_matrix_does_not_crash(self):
        assert check_feature_quality(pd.DataFrame(columns=FEATURE_COLUMNS)) is not None


class TestRetrainAbortsOnDegradedData:
    def test_degraded_build_does_not_train_or_save_state(self, tmp_path, monkeypatch):
        import train as train_mod

        degraded = _matrix(n=200, fallback_rows=200)
        y = pd.Series([1, 0] * 100, name="home_win")
        monkeypatch.setattr(train_mod, "get_or_build_season_features",
                            lambda season, force_rebuild: (degraded, y))
        called = {"train": False, "state": False}
        monkeypatch.setattr(train_mod, "train_models",
                            lambda *a, **k: called.__setitem__("train", True))
        monkeypatch.setattr(train_mod, "save_training_state",
                            lambda s: called.__setitem__("state", True))
        monkeypatch.setattr(train_mod, "load_training_state", lambda: {})

        result = train_mod.run_incremental_retrain(force=True, current_year=2026)

        assert called["train"] is False, "trained a model on league-average constants"
        assert called["state"] is False, "recorded a training run that never happened"
        # Both weekly-retrain.yml and the retrain fallback in daily-predict.yml
        # treat a zero exit as a successful retrain. A silent abort would leave
        # the pipeline predicting with a stale/schema-mismatched model while CI
        # reports green, so the abort MUST be reportable as a failure.
        assert result is False

    def test_override_flag_allows_training(self, tmp_path, monkeypatch):
        import train as train_mod

        degraded = _matrix(n=200, fallback_rows=200)
        y = pd.Series([1, 0] * 100, name="home_win")
        monkeypatch.setattr(train_mod, "get_or_build_season_features",
                            lambda season, force_rebuild: (degraded, y))
        called = {"train": False}
        monkeypatch.setattr(train_mod, "train_models",
                            lambda *a, **k: called.__setitem__("train", True))
        monkeypatch.setattr(train_mod, "save_training_state", lambda s: None)
        monkeypatch.setattr(train_mod, "load_training_state", lambda: {})

        result = train_mod.run_incremental_retrain(force=True, current_year=2026,
                                                   allow_degraded_data=True)
        assert called["train"] is True
        assert result is True

    def test_clean_build_trains_normally(self, tmp_path, monkeypatch):
        import train as train_mod

        clean = _matrix(n=200, fallback_rows=0)
        y = pd.Series([1, 0] * 100, name="home_win")
        monkeypatch.setattr(train_mod, "get_or_build_season_features",
                            lambda season, force_rebuild: (clean, y))
        called = {"train": False}
        monkeypatch.setattr(train_mod, "train_models",
                            lambda *a, **k: called.__setitem__("train", True))
        monkeypatch.setattr(train_mod, "save_training_state", lambda s: None)
        monkeypatch.setattr(train_mod, "load_training_state", lambda: {})

        result = train_mod.run_incremental_retrain(force=True, current_year=2026)
        assert called["train"] is True
        assert result is True


class TestRetrainExitStatus:
    """A data-quality abort must surface as a non-zero exit status.

    .github/workflows/weekly-retrain.yml runs `python main.py --retrain`
    directly, and daily-predict.yml uses it as the `|| python main.py --retrain`
    fallback behind a missing-model / schema-change gate. Both read the exit
    code, so returning normally would report a green retrain that never ran.
    """

    def test_no_training_data_reports_failure(self, monkeypatch):
        import train as train_mod
        monkeypatch.setattr(train_mod, "get_or_build_season_features",
                            lambda season, force_rebuild: (
                                pd.DataFrame(columns=FEATURE_COLUMNS),
                                pd.Series([], dtype=int, name="home_win")))
        monkeypatch.setattr(train_mod, "load_training_state", lambda: {})
        assert train_mod.run_incremental_retrain(force=True, current_year=2026) is False

    def test_main_retrain_exits_non_zero_on_abort(self, monkeypatch):
        import main as main_mod
        monkeypatch.setattr(main_mod, "run_retrain", lambda **kw: False)
        monkeypatch.setattr(main_mod, "init_db", lambda: None)
        monkeypatch.setattr(sys, "argv", ["main.py", "--retrain"])
        with pytest.raises(SystemExit) as exc:
            main_mod.main()
        assert exc.value.code == 1

    def test_main_retrain_exits_zero_on_success(self, monkeypatch):
        import main as main_mod
        monkeypatch.setattr(main_mod, "run_retrain", lambda **kw: True)
        monkeypatch.setattr(main_mod, "init_db", lambda: None)
        monkeypatch.setattr(sys, "argv", ["main.py", "--retrain"])
        main_mod.main()  # must not raise SystemExit

    def test_run_retrain_forwards_the_status(self, monkeypatch):
        import main as main_mod
        import train as train_mod
        monkeypatch.setattr(train_mod, "run_incremental_retrain", lambda **kw: False)
        assert main_mod.run_retrain() is False
        monkeypatch.setattr(train_mod, "run_incremental_retrain", lambda **kw: True)
        assert main_mod.run_retrain() is True
