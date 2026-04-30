"""Tests for evaluate.py — EV, edge, bet sizing, and pick filtering."""

import pytest
from evaluate import (
    calculate_ev,
    calculate_edge,
    size_bet,
    filter_positive_ev,
    format_picks,
    format_stats,
)


# ---------------------------------------------------------------------------
# calculate_ev
# ---------------------------------------------------------------------------

class TestCalculateEV:
    def test_positive_ev_underdog(self):
        # Model says 45% win chance; sportsbook implies 40% (+150).
        # Profit per unit on +150 = 1.5; loss = -1.
        # EV = 0.45*1.5 - 0.55*1 = 0.675 - 0.55 = 0.125
        ev = calculate_ev(model_prob=0.45, implied_prob=0.40, american_odds=150)
        assert ev == pytest.approx(0.125, abs=1e-4)

    def test_negative_ev(self):
        # Model says 40%, sportsbook implies 60% (-150). Should be negative EV.
        ev = calculate_ev(model_prob=0.40, implied_prob=0.60, american_odds=-150)
        assert ev < 0

    def test_zero_edge_is_near_zero_ev(self):
        # When model prob == implied prob there should be no edge / near-zero EV.
        ev = calculate_ev(model_prob=0.50, implied_prob=0.50, american_odds=100)
        assert abs(ev) < 0.01

    def test_heavy_favorite(self):
        # -300 favorite; decimal = 1.333; profit = 0.333
        ev = calculate_ev(model_prob=0.80, implied_prob=0.75, american_odds=-300)
        # EV = 0.80*0.333 - 0.20*1 = 0.266 - 0.20 = 0.066
        assert ev == pytest.approx(0.0667, abs=1e-3)

    def test_return_is_rounded_to_4dp(self):
        ev = calculate_ev(0.55, 0.50, 100)
        assert ev == round(ev, 4)


# ---------------------------------------------------------------------------
# calculate_edge
# ---------------------------------------------------------------------------

class TestCalculateEdge:
    def test_positive_edge(self):
        assert calculate_edge(0.55, 0.50) == pytest.approx(0.05, abs=1e-4)

    def test_negative_edge(self):
        assert calculate_edge(0.40, 0.60) == pytest.approx(-0.20, abs=1e-4)

    def test_zero_edge(self):
        assert calculate_edge(0.50, 0.50) == 0.0

    def test_rounded_to_4dp(self):
        edge = calculate_edge(0.5123456, 0.5000000)
        assert edge == round(edge, 4)


# ---------------------------------------------------------------------------
# size_bet
# ---------------------------------------------------------------------------

class TestSizeBet:
    def test_returns_float(self):
        result = size_bet(0.55, -110)
        assert isinstance(result, float)

    def test_capped_at_max(self):
        # Strong edge should be capped at MAX_BET_UNITS (3.0)
        result = size_bet(0.90, +300)
        assert result <= 3.0

    def test_floored_at_min(self):
        # Negative Kelly (model prob < implied prob) → MIN_BET_UNITS
        result = size_bet(0.40, -200)
        assert result >= 0.5

    def test_exact_tier_boundaries(self):
        # Higher probability at same odds → more units
        low  = size_bet(0.52, -110)
        high = size_bet(0.65, -110)
        assert high >= low

    def test_below_threshold_still_returns_value(self):
        # size_bet should not crash on any valid inputs
        result = size_bet(0.50, +100)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# filter_positive_ev
# ---------------------------------------------------------------------------

def _make_game(home_prob=0.55, home_odds=-110, away_odds=-110, model_name="xgboost"):
    return {
        "game_date": "2026-04-02",
        "home_team": "NYY",
        "away_team": "BOS",
        "home_pitcher_name": "Cole",
        "away_pitcher_name": "Sale",
        "model_prob": home_prob,
        "home_odds": home_odds,
        "away_odds": away_odds,
        "model_name": model_name,
    }


class TestFilterPositiveEV:
    def test_no_picks_when_no_edge(self):
        # -110 on both sides implies ~52.4% each; model at 52% → no edge
        game = _make_game(home_prob=0.52, home_odds=-110, away_odds=-110)
        picks = filter_positive_ev([game])
        assert picks == []

    def test_home_pick_generated(self):
        # Strong model edge on home side
        game = _make_game(home_prob=0.65, home_odds=+130, away_odds=-150)
        picks = filter_positive_ev([game])
        home_picks = [p for p in picks if p["pick_side"] == "Home"]
        assert len(home_picks) == 1
        assert home_picks[0]["pick"] == "NYY"

    def test_away_pick_generated(self):
        # Strong model edge on away side
        game = _make_game(home_prob=0.35, home_odds=-150, away_odds=+130)
        picks = filter_positive_ev([game])
        away_picks = [p for p in picks if p["pick_side"] == "Away"]
        assert len(away_picks) == 1
        assert away_picks[0]["pick"] == "BOS"

    def test_picks_sorted_by_ev_desc(self):
        games = [
            _make_game(home_prob=0.60, home_odds=+150, away_odds=-200),  # good edge
            _make_game(home_prob=0.55, home_odds=+120, away_odds=-150),  # smaller edge
        ]
        # Make teams unique
        games[1]["home_team"] = "LAD"
        games[1]["away_team"] = "SF"
        picks = filter_positive_ev(games)
        evs = [p["ev"] for p in picks]
        assert evs == sorted(evs, reverse=True)

    def test_pick_dict_has_required_keys(self):
        game = _make_game(home_prob=0.65, home_odds=+130, away_odds=-150)
        picks = filter_positive_ev([game])
        required = {"date", "home_team", "away_team", "pick", "pick_side",
                    "model_prob", "implied_prob", "ev", "edge", "units",
                    "odds", "model_name", "home_pitcher", "away_pitcher"}
        for p in picks:
            assert required.issubset(p.keys()), f"Missing keys: {required - p.keys()}"

    def test_empty_input(self):
        assert filter_positive_ev([]) == []


# ---------------------------------------------------------------------------
# format_picks / format_stats
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_picks_no_picks(self):
        output = format_picks([])
        assert "No +EV picks" in output

    def test_format_picks_with_picks(self):
        game = _make_game(home_prob=0.65, home_odds=+130, away_odds=-150)
        picks = filter_positive_ev([game])
        output = format_picks(picks)
        assert "NYY" in output
        assert "BOS" in output
        assert "GAME" in output  # header

    def test_format_stats_zero_bets(self):
        stats = {
            "total_bets": 0, "wins": 0, "losses": 0, "pending": 0,
            "total_units_wagered": 0, "total_profit": 0.0,
            "roi_pct": 0.0, "brier_score": None, "win_rate": 0.0,
        }
        output = format_stats(stats)
        assert "LIFETIME PERFORMANCE" in output
        assert "0W - 0L" in output

    def test_format_stats_with_brier(self):
        stats = {
            "total_bets": 10, "wins": 6, "losses": 4, "pending": 2,
            "total_units_wagered": 15.0, "total_profit": 3.5,
            "roi_pct": 23.3, "brier_score": 0.2123, "win_rate": 60.0,
        }
        output = format_stats(stats)
        assert "Brier Score" in output
        assert "0.2123" in output
