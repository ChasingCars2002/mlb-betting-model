"""Tests for score.py — analytical run estimates."""

import pytest

from score import (
    predict_game_scores, _BASE_RUNS, _LEAGUE_AVG_FIP, _LEAGUE_AVG_BULLPEN,
    _LEAGUE_AVG_OPS,
)


def _features(park=1.0, **overrides):
    """A perfectly league-average matchup.

    These MUST be score.py's own league reference constants. The fixture used
    to hard-code xFIP 4.50 / OPS 0.720, which are not the league means of those
    feature slots — the same mismatch that biased every projection ~0.5 runs
    under the market line.
    """
    base = {
        "home_p_xFIP_season": _LEAGUE_AVG_FIP, "away_p_xFIP_season": _LEAGUE_AVG_FIP,
        "home_bullpen_fip": _LEAGUE_AVG_BULLPEN, "away_bullpen_fip": _LEAGUE_AVG_BULLPEN,
        "home_hit_ops": _LEAGUE_AVG_OPS, "away_hit_ops": _LEAGUE_AVG_OPS,
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


# ---------------------------------------------------------------------------
# Run-scale calibration (regression)
# ---------------------------------------------------------------------------

class TestRunScaleMatchesTheFeatureScale:
    """The "xFIP" slot is FIP built with data._FIP_CONSTANT = 3.10, whose league
    mean is ~4.00 — not the ~4.50 of league ERA. Dividing it by 4.50 scaled
    every projection by ~0.89, so the model sat ~0.5 runs under the market line
    and 67% of all totals picks were Unders (with the Over side running -19%
    ROI on the picks that did clear).
    """

    MARKET_TOTAL_RANGE = (8.0, 9.5)  # typical MLB posted totals

    def test_league_average_game_lands_on_a_realistic_market_total(self):
        total = predict_game_scores(_features())["predicted_total"]
        lo, hi = self.MARKET_TOTAL_RANGE
        assert lo <= total <= hi, (
            f"league-average projection {total} is outside the range books "
            f"actually post; the reference constants are off-scale"
        )

    def test_reference_constants_are_on_the_feature_scale(self):
        from unittest.mock import patch
        import data as data_mod
        # The FIP slot's league mean must sit near the FIP constant plus the
        # usual ~0.9 defense-independent component, nowhere near league ERA.
        assert data_mod._FIP_CONSTANT < _LEAGUE_AVG_FIP < data_mod._FIP_CONSTANT + 1.5
        # The OPS reference must be the same league-average value that
        # data.get_team_hitting_splits substitutes when a split is
        # unavailable; if the two drift apart the under-bias reappears.
        with patch.object(data_mod, "_team_id_from_abbrev", return_value=None):
            league_default = data_mod.get_team_hitting_splits("__none__", "R", season=2026)
        assert league_default["ops"] == _LEAGUE_AVG_OPS

    def test_bullpen_is_actually_used(self):
        """The bullpen throws ~45% of innings. Ignoring it (the old behaviour)
        made the projection swing on the starter alone."""
        good_pen = predict_game_scores(_features(away_bullpen_fip=3.00))
        bad_pen = predict_game_scores(_features(away_bullpen_fip=5.50))
        assert bad_pen["predicted_home_score"] > good_pen["predicted_home_score"]

    def test_starter_outweighs_bullpen(self):
        starter_swing = (
            predict_game_scores(_features(away_p_xFIP_season=5.5))["predicted_home_score"]
            - predict_game_scores(_features(away_p_xFIP_season=3.0))["predicted_home_score"]
        )
        pen_swing = (
            predict_game_scores(_features(away_bullpen_fip=5.5))["predicted_home_score"]
            - predict_game_scores(_features(away_bullpen_fip=3.0))["predicted_home_score"]
        )
        assert starter_swing > pen_swing > 0

    def test_extreme_starter_stays_in_a_plausible_band(self):
        """A raw ratio treated a 33%-better pitcher as suppressing 33% of runs.
        Damping keeps even extreme inputs inside a believable range."""
        vs_ace = predict_game_scores(_features(away_p_xFIP_season=2.00))
        vs_scrub = predict_game_scores(_features(away_p_xFIP_season=7.00))
        assert 2.5 < vs_ace["predicted_home_score"]
        assert vs_scrub["predicted_home_score"] < 8.0

    def test_no_systematic_under_bias_across_a_realistic_slate(self):
        """Sweep plausible matchups; the mean projection must straddle a
        typical 8.5 line rather than sitting persistently beneath it."""
        totals = []
        for starter in (3.0, 3.5, 4.0, 4.5, 5.0):
            for ops in (0.690, 0.720, 0.740, 0.760, 0.790):
                totals.append(predict_game_scores(_features(
                    home_p_xFIP_season=starter, away_p_xFIP_season=starter,
                    home_hit_ops=ops, away_hit_ops=ops,
                ))["predicted_total"])
        mean_total = sum(totals) / len(totals)
        assert 8.2 <= mean_total <= 9.2, f"mean projection {mean_total:.2f}"
