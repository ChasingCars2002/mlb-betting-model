"""Tests for odds.py — conversion functions and consensus averaging."""

import pytest
from odds import (
    american_to_implied_prob,
    american_to_decimal,
    decimal_to_implied_prob,
    implied_prob_to_american,
    devig_two_way,
    match_odds_to_games,
)


# ---------------------------------------------------------------------------
# devig_two_way
# ---------------------------------------------------------------------------

class TestDevigTwoWay:
    def test_sums_to_one(self):
        h, a = devig_two_way(+120, -140)
        assert h + a == pytest.approx(1.0, abs=1e-9)

    def test_symmetric_market(self):
        # -110 / -110 is a perfectly symmetric market → 50/50 after de-vig
        h, a = devig_two_way(-110, -110)
        assert h == pytest.approx(0.5, abs=1e-6)
        assert a == pytest.approx(0.5, abs=1e-6)

    def test_strips_vig(self):
        # Raw implied probs sum to >1; the no-vig home prob must be lower than raw.
        raw_home = american_to_implied_prob(-140)
        h, _ = devig_two_way(-140, +120)
        assert h < raw_home
        assert 0 < h < 1

    def test_favorite_has_higher_prob(self):
        # Home is the favorite (-200) → its no-vig prob should exceed the dog's.
        h, a = devig_two_way(-200, +170)
        assert h > a


# ---------------------------------------------------------------------------
# american_to_implied_prob
# ---------------------------------------------------------------------------

class TestAmericanToImpliedProb:
    def test_plus_150(self):
        assert american_to_implied_prob(150) == pytest.approx(0.4, abs=1e-4)

    def test_minus_150(self):
        assert american_to_implied_prob(-150) == pytest.approx(0.6, abs=1e-4)

    def test_even_money(self):
        assert american_to_implied_prob(100) == pytest.approx(0.5, abs=1e-4)

    def test_heavy_favourite(self):
        # -300: 300/400 = 0.75
        assert american_to_implied_prob(-300) == pytest.approx(0.75, abs=1e-4)

    def test_big_underdog(self):
        # +300: 100/400 = 0.25
        assert american_to_implied_prob(300) == pytest.approx(0.25, abs=1e-4)

    def test_result_in_zero_one(self):
        for odds in [-500, -200, -110, 100, 110, 200, 500]:
            prob = american_to_implied_prob(odds)
            assert 0 < prob < 1


# ---------------------------------------------------------------------------
# implied_prob_to_american
# ---------------------------------------------------------------------------

class TestImpliedProbToAmerican:
    def test_round_trip_underdog(self):
        # +150 → 0.4 → +150
        prob = american_to_implied_prob(150)
        back = implied_prob_to_american(prob)
        assert back == pytest.approx(150, abs=1)

    def test_round_trip_favourite(self):
        # -150 → 0.6 → -150
        prob = american_to_implied_prob(-150)
        back = implied_prob_to_american(prob)
        assert back == pytest.approx(-150, abs=1)

    def test_round_trip_even(self):
        # +100 and -100 are both equivalent even money (prob = 0.5).
        # The conversion picks one sign; just verify magnitude is 100.
        prob = american_to_implied_prob(100)
        back = implied_prob_to_american(prob)
        assert abs(back) == pytest.approx(100, abs=1)

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError):
            implied_prob_to_american(0.0)
        with pytest.raises(ValueError):
            implied_prob_to_american(1.0)
        with pytest.raises(ValueError):
            implied_prob_to_american(1.5)


# ---------------------------------------------------------------------------
# Consensus odds averaging is probability-correct
# ---------------------------------------------------------------------------

class TestOddsAveraging:
    def test_average_through_prob_space(self):
        # Two books: +150 (0.40) and +120 (0.4545)
        # Arithmetic average of American: (+150+120)/2 = +135 → 0.426
        # Prob-space average: (0.40+0.4545)/2 = 0.4273 → +134
        # The two should differ, confirming we're going through prob space.
        p1 = american_to_implied_prob(150)   # 0.40
        p2 = american_to_implied_prob(120)   # ~0.4545
        avg_prob = (p1 + p2) / 2             # ~0.4273
        result = implied_prob_to_american(avg_prob)
        # Should be close to +134, NOT simple arithmetic average +135
        assert 130 <= result <= 138

    def test_symmetric_averaging(self):
        # If both books agree exactly, consensus should equal the original
        p = american_to_implied_prob(150)
        avg = (p + p) / 2
        assert implied_prob_to_american(avg) == pytest.approx(150, abs=1)


# ---------------------------------------------------------------------------
# american_to_decimal
# ---------------------------------------------------------------------------

class TestAmericanToDecimal:
    def test_plus_150(self):
        assert american_to_decimal(150) == pytest.approx(2.50, abs=1e-4)

    def test_minus_150(self):
        assert american_to_decimal(-150) == pytest.approx(1.6667, abs=1e-3)

    def test_even_money(self):
        assert american_to_decimal(100) == pytest.approx(2.00, abs=1e-4)

    def test_decimal_always_gt_1(self):
        for odds in [-500, -200, -110, 100, 200, 500]:
            assert american_to_decimal(odds) > 1.0


# ---------------------------------------------------------------------------
# decimal_to_implied_prob
# ---------------------------------------------------------------------------

class TestDecimalToImpliedProb:
    def test_2_50(self):
        assert decimal_to_implied_prob(2.50) == pytest.approx(0.40, abs=1e-4)

    def test_zero_odds_returns_zero(self):
        assert decimal_to_implied_prob(0) == 0.0

    def test_negative_odds_returns_zero(self):
        assert decimal_to_implied_prob(-1) == 0.0


# ---------------------------------------------------------------------------
# match_odds_to_games
# ---------------------------------------------------------------------------

class TestMatchOddsToGames:
    def _odds(self, home, away):
        return {"home_team": home, "away_team": away, "home_odds": -110, "away_odds": -110}

    def _game(self, home, away):
        return {"home_team": home, "away_team": away, "game_date": "2026-04-02"}

    def test_exact_match(self):
        odds = [self._odds("NYY", "BOS")]
        games = [self._game("NYY", "BOS")]
        result = match_odds_to_games(odds, games)
        assert len(result) == 1
        assert result[0]["home_odds"] == -110

    def test_unmatched_game_excluded(self):
        odds = [self._odds("NYY", "BOS")]
        games = [self._game("LAD", "SF")]
        result = match_odds_to_games(odds, games)
        assert result == []

    def test_partial_match(self):
        odds = [self._odds("NYY", "BOS"), self._odds("LAD", "SF")]
        games = [self._game("NYY", "BOS"), self._game("HOU", "TEX")]
        result = match_odds_to_games(odds, games)
        assert len(result) == 1
        assert result[0]["home_team"] == "NYY"

    def test_merged_keys(self):
        odds = [self._odds("NYY", "BOS")]
        games = [self._game("NYY", "BOS")]
        result = match_odds_to_games(odds, games)
        assert "game_date" in result[0]
        assert "home_odds" in result[0]
