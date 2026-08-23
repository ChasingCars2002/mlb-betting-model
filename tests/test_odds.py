"""Tests for odds.py — conversion functions and consensus averaging."""

from unittest.mock import patch, MagicMock

import pytest
import odds as odds_mod
from odds import (
    american_to_implied_prob,
    american_to_decimal,
    decimal_to_implied_prob,
    implied_prob_to_american,
    devig_two_way,
    match_odds_to_games,
    fetch_live_odds,
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


# ---------------------------------------------------------------------------
# fetch_live_odds — h2h + totals parsing
# ---------------------------------------------------------------------------

class TestFetchLiveOddsTotals:
    def _event(self):
        return {
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "commence_time": "2026-04-02T23:05:00Z",
            "bookmakers": [{
                "key": "draftkings",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "New York Yankees", "price": -140},
                        {"name": "Boston Red Sox", "price": +120},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": -110, "point": 8.5},
                        {"name": "Under", "price": -110, "point": 8.5},
                    ]},
                ],
            }],
        }

    def _fetch(self, raw):
        resp = MagicMock()
        resp.json.return_value = raw
        resp.raise_for_status.return_value = None
        with patch.object(odds_mod, "ODDS_API_KEY", "test-key"), \
             patch.object(odds_mod.requests, "get", return_value=resp):
            return fetch_live_odds()

    def test_parses_totals_market(self):
        result = self._fetch([self._event()])
        assert len(result) == 1
        rec = result[0]
        assert rec["home_team"] == "NYY" and rec["away_team"] == "BOS"
        assert rec["total_line"] == 8.5
        assert rec["over_odds"] is not None
        assert rec["under_odds"] is not None

    def test_no_totals_market_leaves_none(self):
        event = self._event()
        # Drop the totals market, keep only h2h.
        event["bookmakers"][0]["markets"] = [event["bookmakers"][0]["markets"][0]]
        rec = self._fetch([event])[0]
        assert rec["total_line"] is None
        assert rec["over_odds"] is None
        assert rec["under_odds"] is None
        # Moneyline still parsed.
        assert "home_odds" in rec


# ---------------------------------------------------------------------------
# match_odds_to_games — date scoping (regression)
# ---------------------------------------------------------------------------

class TestMatchOddsSeriesScoping:
    """The-Odds-API returns every upcoming event, and MLB plays the same
    matchup on consecutive days. Keying the lookup on (home, away) alone let
    the LAST event in a series overwrite the earlier ones, so today's game
    could be priced off tomorrow's line and tomorrow's starting pitcher.
    """

    def _odds(self, home, away, commence, home_odds=-110):
        return {"home_team": home, "away_team": away,
                "home_odds": home_odds, "away_odds": -110,
                "commence_time": commence}

    def _game(self, home, away, game_date):
        return {"home_team": home, "away_team": away, "game_date": game_date}

    def test_series_game_does_not_clobber_todays_line(self):
        # A three-game series: only the 2026-04-02 game should match.
        series = [
            self._odds("NYY", "BOS", "2026-04-02T23:05:00Z", home_odds=-110),
            self._odds("NYY", "BOS", "2026-04-03T23:05:00Z", home_odds=-200),
            self._odds("NYY", "BOS", "2026-04-04T23:05:00Z", home_odds=+150),
        ]
        result = match_odds_to_games(series, [self._game("NYY", "BOS", "2026-04-02")])
        assert len(result) == 1
        assert result[0]["home_odds"] == -110

    def test_matches_middle_game_of_series(self):
        series = [
            self._odds("NYY", "BOS", "2026-04-02T23:05:00Z", home_odds=-110),
            self._odds("NYY", "BOS", "2026-04-03T23:05:00Z", home_odds=-200),
        ]
        result = match_odds_to_games(series, [self._game("NYY", "BOS", "2026-04-03")])
        assert result[0]["home_odds"] == -200

    def test_game_with_no_event_on_that_date_is_excluded(self):
        odds = [self._odds("NYY", "BOS", "2026-04-05T23:05:00Z")]
        result = match_odds_to_games(odds, [self._game("NYY", "BOS", "2026-04-02")])
        assert result == []

    def test_late_night_game_stays_on_its_eastern_date(self):
        # 22:10 ET first pitch is 02:10Z the NEXT day. Bucketing on the raw UTC
        # date would push it onto tomorrow's slate and drop the match.
        odds = [self._odds("LAD", "SF", "2026-04-03T02:10:00Z")]
        result = match_odds_to_games(odds, [self._game("LAD", "SF", "2026-04-02")])
        assert len(result) == 1

    def test_missing_commence_time_still_matches(self):
        # Back-compat: an event with no usable commence_time falls back to a
        # team-pair match rather than being silently dropped.
        odds = [{"home_team": "NYY", "away_team": "BOS",
                 "home_odds": -125, "away_odds": +105}]
        result = match_odds_to_games(odds, [self._game("NYY", "BOS", "2026-04-02")])
        assert len(result) == 1
        assert result[0]["home_odds"] == -125

    def test_doubleheader_resolves_to_first_game(self):
        odds = [
            self._odds("NYY", "BOS", "2026-04-02T21:05:00Z", home_odds=+120),
            self._odds("NYY", "BOS", "2026-04-02T17:05:00Z", home_odds=-140),
        ]
        result = match_odds_to_games(odds, [self._game("NYY", "BOS", "2026-04-02")])
        assert len(result) == 1
        assert result[0]["home_odds"] == -140


# ---------------------------------------------------------------------------
# fetch_live_odds — totals must be bucketed by line (regression)
# ---------------------------------------------------------------------------

class TestTotalsLineBucketing:
    """Prices are only meaningful attached to their own line. Averaging every
    Over price across books posting 8.5 AND 9.0, then pairing the result with
    the average line (8.75), produced a quote belonging to no real market.
    """

    def _book(self, name, line, over_price, under_price):
        return {"key": name, "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "price": over_price, "point": line},
            {"name": "Under", "price": under_price, "point": line},
        ]}]}

    def _h2h(self, name):
        return {"key": name, "markets": [{"key": "h2h", "outcomes": [
            {"name": "New York Yankees", "price": -130},
            {"name": "Boston Red Sox", "price": +110},
        ]}]}

    def _fetch(self, bookmakers):
        event = {"home_team": "New York Yankees", "away_team": "Boston Red Sox",
                 "commence_time": "2026-04-02T23:05:00Z", "bookmakers": bookmakers}
        resp = MagicMock()
        resp.json.return_value = [event]
        resp.raise_for_status.return_value = None
        with patch.object(odds_mod, "ODDS_API_KEY", "test-key"), \
             patch.object(odds_mod.requests, "get", return_value=resp):
            return fetch_live_odds()[0]

    def test_consensus_line_is_the_most_posted_not_the_average(self):
        rec = self._fetch([
            self._h2h("a"),
            self._book("a", 8.5, -110, -110),
            self._book("b", 8.5, -105, -115),
            self._book("c", 9.0, +100, -120),
        ])
        # Two books at 8.5, one at 9.0 → 8.5 wins. The old code averaged the
        # points to 8.67 and rounded to 8.7, a line nobody offered.
        assert rec["total_line"] == 8.5
        assert rec["num_books_at_line"] == 2

    def test_prices_come_only_from_the_consensus_line(self):
        rec = self._fetch([
            self._h2h("a"),
            self._book("a", 8.5, -110, -110),
            self._book("b", 8.5, -110, -110),
            self._book("c", 9.0, +180, -220),  # far-off line, extreme prices
        ])
        # The 9.0 book's prices must not leak into the 8.5 quote.
        assert rec["over_odds"] == -110
        assert rec["under_odds"] == -110

    def test_one_sided_line_is_ignored(self):
        rec = self._fetch([
            self._h2h("a"),
            {"key": "b", "markets": [{"key": "totals", "outcomes": [
                {"name": "Over", "price": -110, "point": 9.5},
            ]}]},
            self._book("c", 8.5, -105, -115),
        ])
        assert rec["total_line"] == 8.5

    def test_no_totals_market_leaves_fields_none(self):
        rec = self._fetch([self._h2h("a")])
        assert rec["total_line"] is None
        assert rec["over_odds"] is None
