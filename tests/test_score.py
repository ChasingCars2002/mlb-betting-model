"""Tests for score.py — analytical run estimates."""

import pytest

from score import predict_game_scores, _BASE_RUNS


def _features(park=1.0, **overrides):
    base = {
        "home_p_xFIP_season": 4.50, "away_p_xFIP_season": 4.50,
        "home_hit_ops": 0.720, "away_hit_ops": 0.720,
        "park_factor": park,
    }
    base.update(overrides)
    return base


class TestPredictGameScores:
    def test_league_average_matchup(self):
        scores = predict_game_scores(_features())
        # Both sides near base runs; home gets the small home-field bump
        assert scores["predicted_away_score"] == pytest.approx(_BASE_RUNS, abs=0.01)
        assert scores["predicted_home_score"] > scores["predicted_away_score"]
        assert scores["predicted_total"] == pytest.approx(
            scores["predicted_home_score"] + scores["predicted_away_score"], abs=0.02)

    def test_park_factor_boosts_both_teams(self):
        # Both teams hit in the same stadium — a hitter's park must raise
        # BOTH run estimates, not just the home team's.
        neutral = predict_game_scores(_features(park=1.0))
        coors = predict_game_scores(_features(park=1.22))
        assert coors["predicted_home_score"] > neutral["predicted_home_score"]
        assert coors["predicted_away_score"] > neutral["predicted_away_score"]
        assert coors["predicted_away_score"] == pytest.approx(
            neutral["predicted_away_score"] * 1.22, abs=0.05)

    def test_better_opposing_pitcher_lowers_runs(self):
        vs_ace = predict_game_scores(_features(away_p_xFIP_season=2.80))
        vs_avg = predict_game_scores(_features(away_p_xFIP_season=4.50))
        assert vs_ace["predicted_home_score"] < vs_avg["predicted_home_score"]

    def test_missing_features_fall_back_to_league_average(self):
        scores = predict_game_scores({})
        assert scores["predicted_total"] > 0
