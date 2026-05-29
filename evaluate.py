"""EV calculation, bet sizing, and pick filtering."""

import logging

from config import (
    EV_THRESHOLD,
    KELLY_SCALE,
    MIN_BET_UNITS,
    MAX_BET_UNITS,
    MARKET_BLEND_WEIGHT,
    MAX_RAW_DISAGREEMENT,
)
from odds import american_to_implied_prob, american_to_decimal, devig_two_way

logger = logging.getLogger(__name__)


def calculate_ev(model_prob: float, implied_prob: float, american_odds: int) -> float:
    """Calculate Expected Value for a moneyline bet.

    EV = (model_win_prob * potential_profit) - (model_loss_prob * wager)

    For a $1 wager:
        potential_profit = decimal_odds - 1
        EV = (model_prob * profit) - ((1 - model_prob) * 1)

    Args:
        model_prob: Model's predicted probability of this outcome winning.
        implied_prob: Sportsbook's implied probability from the odds.
        american_odds: American moneyline odds (e.g., +150 or -150).

    Returns:
        EV as a fraction of the wager (e.g., 0.05 = 5% EV).
    """
    decimal_odds = american_to_decimal(american_odds)
    profit_per_unit = decimal_odds - 1.0
    ev = (model_prob * profit_per_unit) - ((1 - model_prob) * 1.0)
    return round(ev, 4)


def calculate_edge(model_prob: float, implied_prob: float) -> float:
    """Calculate the edge (model prob - implied prob).

    Positive edge means the model thinks the bet is +EV.
    """
    return round(model_prob - implied_prob, 4)


def size_bet(model_prob: float, american_odds: int) -> float:
    """Determine bet size using half-Kelly criterion.

    Kelly fraction = edge / (decimal_odds - 1)
    Half-Kelly = Kelly * 0.5  (reduces variance while preserving growth)

    Result is capped at 3.0 units. Minimum is the raw half-Kelly value
    (no artificial floor) so small-edge bets are sized proportionally small.
    Returned as a float (e.g., 0.3u or 1.5u).

    Args:
        model_prob: Model's predicted win probability for this side.
        american_odds: American moneyline odds for this bet.
    """
    decimal_odds = american_to_decimal(american_odds)
    if decimal_odds <= 1.0:
        return 0.5
    # Kelly: f* = (model_prob * (decimal_odds - 1) - (1 - model_prob)) / (decimal_odds - 1)
    #           = edge / (decimal_odds - 1)
    kelly = (model_prob * (decimal_odds - 1) - (1 - model_prob)) / (decimal_odds - 1)
    half_kelly = kelly * 0.5
    return round(max(MIN_BET_UNITS, min(MAX_BET_UNITS, half_kelly * KELLY_SCALE)), 2)


def blend_with_market(model_prob: float, no_vig_prob: float,
                      weight: float = MARKET_BLEND_WEIGHT) -> float:
    """Shrink the model probability toward the de-vigged market consensus.

    blended = weight * market + (1 - weight) * model

    A higher `weight` trusts the (sharp) market more. This is the primary
    defense against adverse selection: the raw model overrates the side it
    picks, so betting its un-shrunk probability systematically loses.
    """
    return weight * no_vig_prob + (1.0 - weight) * model_prob


def filter_positive_ev(games_with_predictions: list[dict]) -> list[dict]:
    """Filter games to +EV picks using a market-blended, de-vigged probability.

    For each side we:
      1. De-vig the book consensus to a true probability (sums to 1.0).
      2. Skip the side if the raw model disagrees with that no-vig market by
         more than MAX_RAW_DISAGREEMENT (almost always model error).
      3. Blend the model toward the market (MARKET_BLEND_WEIGHT).
      4. Bet only if the blended prob is +EV at the offered price AND its edge
         over the no-vig market clears EV_THRESHOLD.

    EV, edge, and bet sizing all use the blended probability, so the recorded
    `model_prob` reflects the probability the wager is actually based on and
    `implied_prob` is the no-vig market estimate.

    Args:
        games_with_predictions: List of game dicts, each must have:
            model_prob, home_odds, away_odds, home_team, away_team.

    Returns:
        List of +EV pick dicts ready for database storage.
    """
    picks = []

    for game in games_with_predictions:
        model_prob_home = game["model_prob"]
        model_prob_away = 1.0 - model_prob_home

        # De-vig the two-way market into true probabilities (sum to 1.0).
        home_novig, away_novig = devig_two_way(game["home_odds"], game["away_odds"])

        # Blend model toward the sharp market.
        blended_home = blend_with_market(model_prob_home, home_novig)
        blended_away = 1.0 - blended_home  # internally consistent with blended_home

        # Check home team bet
        home_raw_gap = abs(model_prob_home - home_novig)
        home_edge = calculate_edge(blended_home, home_novig)
        home_ev = calculate_ev(blended_home, home_novig, game["home_odds"])

        if home_raw_gap <= MAX_RAW_DISAGREEMENT and home_ev > 0 and home_edge >= EV_THRESHOLD:
            picks.append({
                "date": game["game_date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "pick": game["home_team"],
                "pick_side": "Home",
                "model_prob": round(blended_home, 4),
                "implied_prob": round(home_novig, 4),
                "ev": home_ev,
                "edge": home_edge,
                "units": size_bet(blended_home, game["home_odds"]),
                "odds": game["home_odds"],
                "model_name": game.get("model_name", "xgboost"),
                "home_pitcher": game.get("home_pitcher_name", ""),
                "away_pitcher": game.get("away_pitcher_name", ""),
            })

        # Check away team bet
        away_raw_gap = abs(model_prob_away - away_novig)
        away_edge = calculate_edge(blended_away, away_novig)
        away_ev = calculate_ev(blended_away, away_novig, game["away_odds"])

        if away_raw_gap <= MAX_RAW_DISAGREEMENT and away_ev > 0 and away_edge >= EV_THRESHOLD:
            picks.append({
                "date": game["game_date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "pick": game["away_team"],
                "pick_side": "Away",
                "model_prob": round(blended_away, 4),
                "implied_prob": round(away_novig, 4),
                "ev": away_ev,
                "edge": away_edge,
                "units": size_bet(blended_away, game["away_odds"]),
                "odds": game["away_odds"],
                "model_name": game.get("model_name", "xgboost"),
                "home_pitcher": game.get("home_pitcher_name", ""),
                "away_pitcher": game.get("away_pitcher_name", ""),
            })

    # Sort by EV descending
    picks.sort(key=lambda x: x["ev"], reverse=True)
    logger.info("Found %d +EV picks from %d games.", len(picks), len(games_with_predictions))
    return picks


def compute_confidence(edge: float, ev: float) -> int:
    """Return a 1–5 star rating based on edge and EV thresholds.

    5 stars: edge >= 10% AND ev >= 0.20
    4 stars: edge >= 7%  AND ev >= 0.12
    3 stars: edge >= 5%  AND ev >= 0.08
    2 stars: edge >= 3%  AND ev >= 0.04
    1 star:  anything that cleared the EV threshold
    """
    if edge >= 0.10 and ev >= 0.20:
        return 5
    if edge >= 0.07 and ev >= 0.12:
        return 4
    if edge >= 0.05 and ev >= 0.08:
        return 3
    if edge >= 0.03 and ev >= 0.04:
        return 2
    return 1


def format_picks(picks: list[dict]) -> str:
    """Format picks into a console-friendly table string."""
    if not picks:
        return "\n  No +EV picks found today.\n"

    lines = []
    lines.append("")
    lines.append("=" * 85)
    lines.append(f"  {'GAME':<25} {'PICK':<8} {'MODEL':>6} {'IMPLIED':>8} "
                 f"{'EDGE':>6} {'EV':>7} {'UNITS':>5} {'ODDS':>7}")
    lines.append("-" * 85)

    total_units = 0.0
    for p in picks:
        matchup = f"{p['away_team']} @ {p['home_team']}"
        lines.append(
            f"  {matchup:<25} {p['pick']:<8} {p['model_prob']:>5.1%} "
            f"{p['implied_prob']:>7.1%} {p['edge']:>5.1%} "
            f"{p['ev']:>+6.1%} {p['units']:>5.1f}u  {p['odds']:>+7d}"
        )
        total_units += p["units"]

    lines.append("-" * 85)
    lines.append(f"  Total: {len(picks)} picks, {total_units:.1f} units wagered")
    lines.append("=" * 85)
    lines.append("")

    return "\n".join(lines)


def format_stats(stats: dict) -> str:
    """Format ROI stats into a readable summary."""
    lines = []
    lines.append("")
    lines.append("=" * 50)
    lines.append("  LIFETIME PERFORMANCE")
    lines.append("=" * 50)
    lines.append(f"  Total Bets:       {stats['total_bets']}")
    lines.append(f"  Record:           {stats['wins']}W - {stats['losses']}L "
                 f"({stats['win_rate']:.1f}%)")
    lines.append(f"  Pending:          {stats['pending']}")
    lines.append(f"  Units Wagered:    {stats['total_units_wagered']:.1f}")
    lines.append(f"  Total Profit:     {stats['total_profit']:+.2f}u")
    lines.append(f"  ROI:              {stats['roi_pct']:+.2f}%")
    if stats["brier_score"] is not None:
        lines.append(f"  Brier Score:      {stats['brier_score']:.4f}")
    lines.append("=" * 50)
    lines.append("")
    return "\n".join(lines)
