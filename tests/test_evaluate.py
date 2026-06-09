"""Tests for evaluate.py — EV, edge, bet sizing, and pick filtering."""

import pytest
from evaluate import (
    calculate_ev,
    calculate_edge,
    size_bet,
    blend_with_market,
    compute_confidence,
    filter_positive_ev,
    filter_totals_ev,
    total_over_probability,
    format_picks,
    format_stats,
)
from config import TOTALS_MAX_DISAGREEMENT


# ---------------------------------------------------------------------------
# blend_with_market
# ---------------------------------------------------------------------------

class TestBlendWithMarket:
    def test_pure_market(self):
        assert blend_with_market(0.70, 0.40, weight=1.0) == pytest.approx(0.40)

    def test_pure_model(self):
        assert blend_with_market(0.70, 0.40, weight=0.0) == pytest.approx(0.70)

    def test_halfway(self):
        assert blend_with_market(0.70, 0.40, weight=0.5) == pytest.approx(0.55)

    def test_shrinks_toward_market(self):
        # Blended value always lies between model and market
        blended = blend_with_market(0.65, 0.45, weight=0.6)
        assert 0.45 < blended < 0.65


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

# Default: model 0.58 on home, priced +120 / -140. After de-vig the home
# no-vig prob is ~0.438, a ~14-pt disagreement that survives the cap and,
# once blended toward the market, still clears the edge threshold.
def _make_game(home_prob=0.58, home_odds=+120, away_odds=-140, model_name="xgboost"):
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
        # -110 on both sides de-vigs to ~50% each; model at 52% → tiny edge, no bet
        game = _make_game(home_prob=0.52, home_odds=-110, away_odds=-110)
        picks = filter_positive_ev([game])
        assert picks == []

    def test_home_pick_generated(self):
        # Moderate, plausible model edge on the home side
        game = _make_game(home_prob=0.58, home_odds=+120, away_odds=-140)
        picks = filter_positive_ev([game])
        home_picks = [p for p in picks if p["pick_side"] == "Home"]
        assert len(home_picks) == 1
        assert home_picks[0]["pick"] == "NYY"

    def test_away_pick_generated(self):
        # Mirror image: moderate model edge on the away side
        game = _make_game(home_prob=0.42, home_odds=-140, away_odds=+120)
        picks = filter_positive_ev([game])
        away_picks = [p for p in picks if p["pick_side"] == "Away"]
        assert len(away_picks) == 1
        assert away_picks[0]["pick"] == "BOS"

    def test_implausible_disagreement_rejected(self):
        # Model says 0.65 but the market de-vigs to ~0.42 — a >20-pt gap that is
        # almost certainly model error. The cap must reject it.
        game = _make_game(home_prob=0.65, home_odds=+130, away_odds=-150)
        picks = filter_positive_ev([game])
        assert picks == []

    def test_blended_prob_sits_between_model_and_market(self):
        game = _make_game(home_prob=0.58, home_odds=+120, away_odds=-140)
        picks = filter_positive_ev([game])
        home = next(p for p in picks if p["pick_side"] == "Home")
        # Recorded model_prob is the blended value: below the raw 0.58 model
        # prob and above the no-vig market (implied_prob).
        assert home["implied_prob"] < home["model_prob"] < 0.58
        # edge is measured against the no-vig market
        assert home["edge"] == pytest.approx(home["model_prob"] - home["implied_prob"], abs=1e-4)


# ---------------------------------------------------------------------------
# total_over_probability
# ---------------------------------------------------------------------------

class TestTotalOverProbability:
    def test_at_line_is_half(self):
        # Predicted total exactly on the line → coin flip.
        assert total_over_probability(8.5, 8.5) == pytest.approx(0.5, abs=1e-9)

    def test_above_line_favors_over(self):
        assert total_over_probability(10.0, 8.5) > 0.5

    def test_below_line_favors_under(self):
        assert total_over_probability(7.0, 8.5) < 0.5

    def test_monotonic_in_prediction(self):
        low  = total_over_probability(8.0, 8.5)
        high = total_over_probability(9.5, 8.5)
        assert high > low

    def test_in_unit_interval(self):
        for pred in (2.0, 8.5, 15.0):
            p = total_over_probability(pred, 8.5)
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# filter_totals_ev
# ---------------------------------------------------------------------------

def _make_totals_game(predicted_total=9.8, total_line=8.5,
                      over_odds=-105, under_odds=-105):
    return {
        "game_date": "2026-04-02",
        "home_team": "NYY",
        "away_team": "BOS",
        "home_pitcher_name": "Cole",
        "away_pitcher_name": "Sale",
        "model_name": "xgboost",
        "predicted_total": predicted_total,
        "predicted_home_runs": predicted_total / 2,
        "predicted_away_runs": predicted_total / 2,
        "total_line": total_line,
        "over_odds": over_odds,
        "under_odds": under_odds,
    }


class TestFilterTotalsEV:
    def test_over_pick_generated(self):
        # Model total well above the line → Over should clear the edge threshold.
        picks = filter_totals_ev([_make_totals_game(predicted_total=10.2, total_line=8.5)])
        overs = [p for p in picks if p["pick"] == "Over"]
        assert len(overs) == 1
        o = overs[0]
        assert o["bet_type"] == "totals"
        assert o["pick_side"] == "Over"
        assert o["listed_total"] == 8.5
        assert o["total_delta"] == pytest.approx(10.2 - 8.5, abs=1e-2)

    def test_under_pick_generated(self):
        picks = filter_totals_ev([_make_totals_game(predicted_total=6.8, total_line=8.5)])
        unders = [p for p in picks if p["pick"] == "Under"]
        assert len(unders) == 1
        assert unders[0]["pick_side"] == "Under"

    def test_no_pick_when_model_matches_line(self):
        picks = filter_totals_ev([_make_totals_game(predicted_total=8.5, total_line=8.5)])
        assert picks == []

    def test_skipped_without_line_or_prediction(self):
        no_line = _make_totals_game()
        no_line["total_line"] = None
        no_pred = _make_totals_game()
        no_pred["predicted_total"] = None
        assert filter_totals_ev([no_line, no_pred]) == []

    def test_implausible_disagreement_rejected(self):
        # A wildly high prediction creates a >MAX_RAW_DISAGREEMENT gap vs the
        # ~50/50 no-vig market and must be rejected as model error.
        picks = filter_totals_ev([_make_totals_game(predicted_total=16.0, total_line=8.5)])
        assert picks == []

    def test_picks_sorted_by_ev_desc(self):
        games = [
            _make_game(home_prob=0.58, home_odds=+120, away_odds=-140),  # larger edge
            _make_game(home_prob=0.56, home_odds=+110, away_odds=-130),  # smaller edge
        ]
        # Make teams unique
        games[1]["home_team"] = "LAD"
        games[1]["away_team"] = "SF"
        picks = filter_positive_ev(games)
        evs = [p["ev"] for p in picks]
        assert evs == sorted(evs, reverse=True)

    def test_pick_dict_has_required_keys(self):
        game = _make_game(home_prob=0.58, home_odds=+120, away_odds=-140)
        picks = filter_positive_ev([game])
        assert picks  # sanity: at least one pick
        required = {"date", "home_team", "away_team", "pick", "pick_side",
                    "model_prob", "implied_prob", "ev", "edge", "units",
                    "odds", "model_name", "home_pitcher", "away_pitcher"}
        for p in picks:
            assert required.issubset(p.keys()), f"Missing keys: {required - p.keys()}"

    def test_empty_input(self):
        assert filter_positive_ev([]) == []


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    """Stars are scaled to the ACHIEVABLE edge band: with blend weight w and
    disagreement cap c, no pick can exceed an edge of (1-w)*c. At the default
    w=0.5 the moneyline band is [0.05, 0.075] and the totals band [0.05, 0.15].
    """

    def test_threshold_edge_is_one_star(self):
        assert compute_confidence(0.05, 0.05) == 1

    def test_max_achievable_edge_is_five_stars(self):
        assert compute_confidence(0.075, 0.10) == 5

    def test_all_tiers_reachable(self):
        edges = (0.051, 0.056, 0.062, 0.068, 0.074)
        stars = [compute_confidence(e, 0.05) for e in edges]
        assert stars == [1, 2, 3, 4, 5]

    def test_monotonic_in_edge(self):
        stars = [compute_confidence(e, 0.05) for e in (0.05, 0.06, 0.07, 0.075)]
        assert stars == sorted(stars)

    def test_totals_band_uses_wider_cap(self):
        # Same edge maps to fewer stars on totals because the band is wider.
        ml = compute_confidence(0.074, 0.05)
        ou = compute_confidence(0.074, 0.05,
                                max_disagreement=TOTALS_MAX_DISAGREEMENT, weight=0.5)
        assert ml == 5
        assert ou < ml
        assert compute_confidence(0.149, 0.2,
                                  max_disagreement=TOTALS_MAX_DISAGREEMENT, weight=0.5) == 5

    def test_never_below_one_or_above_five(self):
        assert compute_confidence(0.0, 0.0) == 1
        assert compute_confidence(0.5, 1.0) == 5


# ---------------------------------------------------------------------------
# format_picks / format_stats
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_picks_no_picks(self):
        output = format_picks([])
        assert "No +EV picks" in output

    def test_format_picks_with_picks(self):
        game = _make_game(home_prob=0.58, home_odds=+120, away_odds=-140)
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
