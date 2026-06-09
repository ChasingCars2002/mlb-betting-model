"""Tests for calibration.py — the self-tuning market blend loop."""

import json
from unittest.mock import patch

import numpy as np
import pytest

import calibration
import database
from config import MARKET_BLEND_WEIGHT

# Bound at import time, BEFORE the autouse conftest fixture stubs them out —
# these tests exercise the real implementations.
_REAL_GET_BLEND_WEIGHT = calibration.get_blend_weight
_REAL_IS_SELF_TUNED = calibration.is_self_tuned


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary database with the full schema (predictions + model_log)."""
    db_file = tmp_path / "test_bets.db"
    with patch("database.DB_PATH", db_file):
        database.init_db()
        yield db_file


@pytest.fixture
def tmp_blend_state(tmp_path, monkeypatch):
    """Point BLEND_STATE_PATH at a temp file, restore the real (unstubbed)
    weight functions, and reset the in-process cache."""
    monkeypatch.setattr(calibration, "get_blend_weight", _REAL_GET_BLEND_WEIGHT)
    monkeypatch.setattr(calibration, "is_self_tuned", _REAL_IS_SELF_TUNED)
    state_path = tmp_path / "blend_state.json"
    with patch("calibration.BLEND_STATE_PATH", state_path):
        calibration._invalidate_cache()
        yield state_path
    calibration._invalidate_cache()


def _log_row(date="2026-06-01", home="NYY", away="BOS",
             model=0.60, market=0.55, home_win=None):
    return {
        "date": date, "home_team": home, "away_team": away,
        "raw_model_prob": model, "market_prob": market,
        "home_odds": -120, "away_odds": 100, "model_name": "xgboost",
        "home_win": home_win,
    }


# ---------------------------------------------------------------------------
# model_log storage and grading
# ---------------------------------------------------------------------------

class TestModelLog:
    def test_save_and_grade(self, tmp_db):
        row = _log_row()
        database.save_model_log([{k: v for k, v in row.items() if k != "home_win"}])

        database.grade_model_log(
            {"BOS @ NYY": {"home_score": 5, "away_score": 3, "winner": "NYY"}},
            for_date="2026-06-01",
        )
        graded = database.get_graded_model_log()
        assert len(graded) == 1
        assert graded[0]["home_win"] == 1
        assert graded[0]["raw_model_prob"] == pytest.approx(0.60)

    def test_upsert_replaces_ungraded_only(self, tmp_db):
        base = {k: v for k, v in _log_row(model=0.60).items() if k != "home_win"}
        database.save_model_log([base])
        # Re-run with fresher odds/prob — should overwrite while ungraded
        database.save_model_log([{**base, "raw_model_prob": 0.62}])

        database.grade_model_log(
            {"BOS @ NYY": {"home_score": 2, "away_score": 7, "winner": "BOS"}},
            for_date="2026-06-01",
        )
        graded = database.get_graded_model_log()
        assert graded[0]["raw_model_prob"] == pytest.approx(0.62)
        assert graded[0]["home_win"] == 0

        # A later save must NOT clobber a graded row
        database.save_model_log([{**base, "raw_model_prob": 0.99}])
        graded = database.get_graded_model_log()
        assert graded[0]["raw_model_prob"] == pytest.approx(0.62)

    def test_pending_dates(self, tmp_db):
        database.save_model_log([
            {k: v for k, v in _log_row(date="2026-06-01").items() if k != "home_win"},
            {k: v for k, v in _log_row(date="2026-06-02", home="LAD", away="SF").items()
             if k != "home_win"},
        ])
        assert sorted(database.get_model_log_dates_pending()) == ["2026-06-01", "2026-06-02"]


# ---------------------------------------------------------------------------
# log_model_predictions
# ---------------------------------------------------------------------------

class TestLogModelPredictions:
    def test_logs_full_slate_and_skips_fallback_probs(self, tmp_db):
        games = [
            {"game_date": "2026-06-01", "home_team": "NYY", "away_team": "BOS",
             "model_prob": 0.61, "home_odds": -130, "away_odds": 110},
            # 0.5 is the "features unavailable" fallback — must be excluded
            {"game_date": "2026-06-01", "home_team": "LAD", "away_team": "SF",
             "model_prob": 0.5, "home_odds": -150, "away_odds": 130},
        ]
        n = calibration.log_model_predictions(games)
        assert n == 1

    def test_market_prob_is_devigged(self, tmp_db):
        games = [{"game_date": "2026-06-01", "home_team": "NYY", "away_team": "BOS",
                  "model_prob": 0.61, "home_odds": -110, "away_odds": -110}]
        calibration.log_model_predictions(games)
        with database.get_connection() as conn:
            row = conn.execute("SELECT market_prob FROM model_log").fetchone()
        # Symmetric -110/-110 de-vigs to exactly 0.5
        assert row["market_prob"] == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# get_blend_weight / update_blend_weight
# ---------------------------------------------------------------------------

class TestBlendWeight:
    def test_default_without_state(self, tmp_blend_state):
        assert calibration.get_blend_weight() == MARKET_BLEND_WEIGHT
        assert calibration.is_self_tuned() is False

    def test_default_below_min_sample(self, tmp_blend_state):
        tmp_blend_state.write_text(json.dumps({"weight": 0.9, "n_games": 10}))
        calibration._invalidate_cache()
        assert calibration.get_blend_weight() == MARKET_BLEND_WEIGHT

    def test_learned_weight_used_and_clamped(self, tmp_blend_state):
        tmp_blend_state.write_text(json.dumps(
            {"weight": 0.99, "n_games": calibration.MIN_CALIBRATION_GAMES}))
        calibration._invalidate_cache()
        assert calibration.get_blend_weight() == calibration.BLEND_WEIGHT_MAX
        assert calibration.is_self_tuned() is True

    def test_update_requires_min_games(self, tmp_db, tmp_blend_state):
        database.save_model_log(
            [{k: v for k, v in _log_row().items() if k != "home_win"}])
        assert calibration.update_blend_weight() is None
        assert not tmp_blend_state.exists()

    def test_update_learns_market_trust_when_model_is_noise(self, tmp_db, tmp_blend_state):
        # Construct a sample where the market is well-calibrated and the model
        # is pure noise: the optimal weight must land at the upper bound.
        rng = np.random.default_rng(42)
        n = 1000
        market = rng.uniform(0.35, 0.65, n)
        outcomes = (rng.uniform(0, 1, n) < market).astype(int)
        model = rng.uniform(0.3, 0.7, n)  # uninformative

        rows = []
        for i in range(n):
            rows.append({
                "date": f"2026-05-{(i % 28) + 1:02d}", "home_team": f"T{i}",
                "away_team": f"U{i}", "raw_model_prob": float(model[i]),
                "market_prob": float(market[i]), "home_odds": -110,
                "away_odds": -110, "model_name": "xgboost",
            })
        database.save_model_log(rows)
        with database.get_connection() as conn:
            for i, out in enumerate(outcomes):
                conn.execute(
                    "UPDATE model_log SET home_win = ? WHERE home_team = ?",
                    (int(out), f"T{i}"),
                )

        state = calibration.update_blend_weight()
        assert state is not None
        assert state["n_games"] == n
        assert state["weight"] >= 0.80  # trusts the market heavily
        assert state["log_loss"] <= state["default_weight_log_loss"]
        # And the persisted state now drives get_blend_weight
        assert calibration.get_blend_weight() == pytest.approx(
            min(calibration.BLEND_WEIGHT_MAX, max(calibration.BLEND_WEIGHT_MIN, state["weight"])))

    def test_update_keeps_model_voice_when_model_is_better(self, tmp_db, tmp_blend_state):
        # Model perfectly calibrated, market biased: optimal weight is low.
        rng = np.random.default_rng(7)
        n = 400
        model = rng.uniform(0.35, 0.65, n)
        outcomes = (rng.uniform(0, 1, n) < model).astype(int)
        market = np.clip(model + 0.12, 0.01, 0.99)  # systematically biased

        rows = []
        for i in range(n):
            rows.append({
                "date": f"2026-05-{(i % 28) + 1:02d}", "home_team": f"T{i}",
                "away_team": f"U{i}", "raw_model_prob": float(model[i]),
                "market_prob": float(market[i]), "home_odds": -110,
                "away_odds": -110, "model_name": "xgboost",
            })
        database.save_model_log(rows)
        with database.get_connection() as conn:
            for i, out in enumerate(outcomes):
                conn.execute(
                    "UPDATE model_log SET home_win = ? WHERE home_team = ?",
                    (int(out), f"T{i}"),
                )

        state = calibration.update_blend_weight()
        assert state is not None
        assert state["weight"] <= 0.5
        # Never goes below the hard floor
        assert state["weight"] >= calibration.BLEND_WEIGHT_MIN
