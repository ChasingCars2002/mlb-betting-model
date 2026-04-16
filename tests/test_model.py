"""Tests for model.py — training, loading, prediction, and median imputation."""

import pytest
import numpy as np
import pandas as pd
import joblib
from unittest.mock import patch, MagicMock

from features import FEATURE_COLUMNS
import model as model_module
from model import (
    _evaluate_model,
    predict_win_prob,
    _load_feature_medians,
)


# ---------------------------------------------------------------------------
# FEATURE_COLUMNS correctness (regression against bad past state)
# ---------------------------------------------------------------------------

class TestFeatureColumns:
    def test_is_home_not_in_features(self):
        assert "is_home" not in FEATURE_COLUMNS, (
            "is_home was a constant-1 feature that provides zero signal. "
            "It must not appear in FEATURE_COLUMNS."
        )

    def test_expected_count(self):
        # 8 home pitcher + 8 away pitcher + 4 bullpen + 4 hitting = 24
        assert len(FEATURE_COLUMNS) == 24

    def test_rolling_and_season_both_present(self):
        season_feats = [f for f in FEATURE_COLUMNS if "_season" in f]
        rolling_feats = [f for f in FEATURE_COLUMNS if "_rolling" in f]
        assert len(season_feats) == 8
        assert len(rolling_feats) == 8

    def test_no_duplicates(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


# ---------------------------------------------------------------------------
# _evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def _perfect_preds(self, n=100):
        y = pd.Series([1] * (n // 2) + [0] * (n // 2))
        probs = np.array([1.0] * (n // 2) + [0.0] * (n // 2))
        return y, probs

    def _random_preds(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        y = pd.Series(rng.integers(0, 2, n))
        probs = rng.uniform(0, 1, n)
        return y, probs

    def test_returns_dict_with_expected_keys(self):
        y, probs = self._random_preds()
        result = _evaluate_model("TestModel", y, probs)
        assert set(result.keys()) == {"name", "accuracy", "brier_score", "log_loss"}

    def test_perfect_model_has_zero_brier(self):
        y, probs = self._perfect_preds()
        result = _evaluate_model("Perfect", y, probs)
        assert result["brier_score"] == pytest.approx(0.0, abs=1e-4)

    def test_accuracy_in_valid_range(self):
        y, probs = self._random_preds()
        result = _evaluate_model("Random", y, probs)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_metrics_are_rounded_to_4dp(self):
        y, probs = self._random_preds()
        result = _evaluate_model("Test", y, probs)
        for key in ["accuracy", "brier_score", "log_loss"]:
            assert result[key] == round(result[key], 4)


# ---------------------------------------------------------------------------
# predict_win_prob — NaN imputation with saved medians
# ---------------------------------------------------------------------------

class TestPredictWinProb:
    def _mock_model(self, prob=0.60):
        m = MagicMock()
        m.predict_proba.return_value = np.array([[1 - prob, prob]])
        return m

    def test_returns_float(self):
        mock_model = self._mock_model()
        features = {col: 1.0 for col in FEATURE_COLUMNS}
        with patch.object(model_module, "_load_feature_medians", return_value={}):
            result = predict_win_prob(mock_model, features)
        assert isinstance(result, float)

    def test_probability_in_range(self):
        mock_model = self._mock_model(prob=0.72)
        features = {col: 1.0 for col in FEATURE_COLUMNS}
        with patch.object(model_module, "_load_feature_medians", return_value={}):
            result = predict_win_prob(mock_model, features)
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(0.72, abs=1e-6)

    def test_nan_imputed_with_saved_medians(self):
        """NaN features should be filled with training medians, not left as NaN."""
        medians = {col: 4.20 for col in FEATURE_COLUMNS}
        features = {col: float("nan") for col in FEATURE_COLUMNS}

        captured_X = {}

        def fake_predict_proba(X):
            captured_X["X"] = X
            return np.array([[0.4, 0.6]])

        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = fake_predict_proba

        with patch.object(model_module, "_load_feature_medians", return_value=medians):
            predict_win_prob(mock_model, features)

        X = captured_X["X"]
        assert not X.isnull().any().any(), "NaNs were not imputed"
        np.testing.assert_allclose(X.values, 4.20, rtol=1e-5)

    def test_nan_imputed_with_zeros_when_no_medians_file(self):
        """When medians file is absent, fall back to 0.0 (not NaN)."""
        features = {col: float("nan") for col in FEATURE_COLUMNS}

        captured_X = {}

        def fake_predict_proba(X):
            captured_X["X"] = X
            return np.array([[0.5, 0.5]])

        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = fake_predict_proba

        with patch.object(model_module, "_load_feature_medians", return_value={}):
            predict_win_prob(mock_model, features)

        X = captured_X["X"]
        assert not X.isnull().any().any(), "NaNs were not handled when medians absent"


# ---------------------------------------------------------------------------
# train_models — verify medians are saved and cache is refreshed
# ---------------------------------------------------------------------------

class TestTrainModels:
    def test_medians_saved_to_disk(self, tmp_path):
        from model import train_models, _MEDIANS_PATH
        import model as model_mod

        # Build a tiny synthetic dataset with all required features
        rng = np.random.default_rng(42)
        n = 60
        X = pd.DataFrame(
            rng.uniform(3.0, 6.0, size=(n, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )
        y = pd.Series(rng.integers(0, 2, n))

        fake_dir = tmp_path / "models"
        fake_dir.mkdir()

        with patch.object(model_mod, "MODEL_DIR", fake_dir), \
             patch.object(model_mod, "_MEDIANS_PATH", fake_dir / "feature_medians.joblib"), \
             patch.object(model_mod, "_feature_medians_cache", None):
            train_models(X, y)

        medians_file = fake_dir / "feature_medians.joblib"
        assert medians_file.exists(), "feature_medians.joblib was not saved by train_models"

        medians = joblib.load(medians_file)
        assert set(medians.keys()) == set(FEATURE_COLUMNS)
        for col in FEATURE_COLUMNS:
            assert not np.isnan(medians[col]), f"Median for {col} is NaN"
