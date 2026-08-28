"""EV calculation, bet sizing, and pick filtering."""

import logging
import math

from config import (
    EV_THRESHOLD,
    KELLY_SCALE,
    MIN_BET_UNITS,
    MAX_BET_UNITS,
    MAX_RAW_DISAGREEMENT,
    TOTALS_MAX_DISAGREEMENT,
    REQUIRE_MEASURED_EDGE,
)
from odds import american_to_implied_prob, american_to_decimal, devig_two_way
import calibration
import totals_calibration

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

    The scaled half-Kelly stake is clamped to [MIN_BET_UNITS, MAX_BET_UNITS]:
    any qualifying pick risks at least the floor, and never more than the cap.
    Returned as a float (e.g., 0.5u or 1.5u).

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
                      weight: float | None = None) -> float:
    """Shrink the model probability toward the de-vigged market consensus.

    blended = weight * market + (1 - weight) * model

    A higher `weight` trusts the (sharp) market more. This is the primary
    defense against adverse selection: the raw model overrates the side it
    picks, so betting its un-shrunk probability systematically loses.

    When `weight` is None the self-tuned weight is used — learned weekly from
    the model's own graded predictions (see calibration.py), falling back to
    the static MARKET_BLEND_WEIGHT until enough games have been graded.
    """
    if weight is None:
        weight = calibration.get_blend_weight()
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

    # Same gate as totals, on the metric a moneyline bet actually depends on.
    # This market has emitted nothing since 2026-06-22 — not by decision, but
    # because the learned blend weight rose until (1 - weight) * cap fell below
    # EV_THRESHOLD and the filter became arithmetically unsatisfiable. Zero
    # picks was the right answer arriving for the wrong reason, and silently.
    gate = moneyline_edge_status()
    if not gate["bettable"]:
        logger.error(
            "MONEYLINE SUPPRESSED (%s): %s No moneyline picks will be emitted.",
            gate["reason"], gate["detail"],
        )
        return picks

    blend_weight = calibration.get_blend_weight()

    for game in games_with_predictions:
        model_prob_home = game["model_prob"]
        model_prob_away = 1.0 - model_prob_home

        # De-vig the two-way market into true probabilities (sum to 1.0).
        home_novig, away_novig = devig_two_way(game["home_odds"], game["away_odds"])

        # Blend model toward the sharp market (weight is self-tuned, see calibration.py).
        blended_home = blend_with_market(model_prob_home, home_novig, weight=blend_weight)
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
                "raw_model_prob": round(model_prob_home, 4),
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
                "raw_model_prob": round(model_prob_away, 4),
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


def total_over_probability(predicted_total: float, line: float,
                           sigma: float | None = None) -> float:
    """Probability the game total finishes Over `line`, per the score model.

    Treats the actual total as Normal(predicted_total, sigma) and returns
    P(total > line) = 1 - Phi((line - predicted_total) / sigma). Uses math.erf
    so there is no scipy/numpy dependency.

    sigma defaults to the learned residual SD (totals_calibration), falling back
    to config.TOTALS_SIGMA before there is a fit. It is the single most
    leveraged number in totals pricing: the edge scales with how far
    P(Over) sits from the market, and understating sigma moves it further out on
    every game at once. The old hard-coded 3.0 was ~40% below the measured
    residual SD, which is how a model with no demonstrated skill still cleared a
    5% edge gate four to eight times a day.
    """
    if sigma is None:
        sigma = totals_calibration.get_totals_sigma()
    if sigma <= 0:
        return 1.0 if predicted_total > line else 0.0
    z = (line - predicted_total) / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 1.0 - cdf


def filter_totals_ev(games_with_odds: list[dict]) -> list[dict]:
    """Filter games to +EV Over/Under picks, mirroring filter_positive_ev.

    For each game with a posted total line and the score model's predicted_total
    we: derive P(Over)/P(Under) from the model, de-vig the book's O/U prices,
    skip sides where the raw model disagrees with the no-vig market by more than
    MAX_RAW_DISAGREEMENT, blend toward the market, and keep sides that are +EV
    and clear EV_THRESHOLD. Returns totals pick dicts ready for the database.
    """
    picks = []

    # A totals bet claims the projection is closer to the truth than the line.
    # When that has not been measured, or has been measured false, there is no
    # bet to make — emitting picks anyway just pays vig on noise.
    gate = totals_calibration.totals_edge_status()
    if not gate["bettable"]:
        logger.error(
            "TOTALS SUPPRESSED (%s): %s No totals picks will be emitted.",
            gate["reason"], gate["detail"],
        )
        return picks

    blend_weight = totals_calibration.get_totals_blend_weight()
    sigma = totals_calibration.get_totals_sigma()

    for game in games_with_odds:
        line = game.get("total_line")
        over_odds = game.get("over_odds")
        under_odds = game.get("under_odds")
        predicted_total = game.get("predicted_total")
        if line is None or over_odds is None or under_odds is None or predicted_total is None:
            continue

        model_over = total_over_probability(predicted_total, line, sigma=sigma)
        model_under = 1.0 - model_over

        over_novig, under_novig = devig_two_way(over_odds, under_odds)
        # Totals get their OWN learned blend weight. The moneyline weight is fit
        # on a gradient-boosted win classifier and says nothing about an
        # analytical run projection; before totals_calibration existed this fell
        # back to the static default with no feedback loop at all.
        blended_over = blend_with_market(model_over, over_novig, weight=blend_weight)
        blended_under = 1.0 - blended_over

        delta = round(predicted_total - line, 2)
        base = {
            "date": game["game_date"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "bet_type": "totals",
            "model_name": game.get("model_name", "xgboost"),
            "home_pitcher": game.get("home_pitcher_name", ""),
            "away_pitcher": game.get("away_pitcher_name", ""),
            "listed_total": line,
            "predicted_total": predicted_total,
            "predicted_home_runs": game.get("predicted_home_runs"),
            "predicted_away_runs": game.get("predicted_away_runs"),
            "total_delta": delta,
        }

        # Over
        over_gap = abs(model_over - over_novig)
        over_edge = calculate_edge(blended_over, over_novig)
        over_ev = calculate_ev(blended_over, over_novig, over_odds)
        if over_gap <= TOTALS_MAX_DISAGREEMENT and over_ev > 0 and over_edge >= EV_THRESHOLD:
            picks.append({
                **base,
                "pick": "Over",
                "pick_side": "Over",
                "model_prob": round(blended_over, 4),
                "raw_model_prob": round(model_over, 4),
                "implied_prob": round(over_novig, 4),
                "ev": over_ev,
                "edge": over_edge,
                "units": size_bet(blended_over, over_odds),
                "odds": over_odds,
            })

        # Under
        under_gap = abs(model_under - under_novig)
        under_edge = calculate_edge(blended_under, under_novig)
        under_ev = calculate_ev(blended_under, under_novig, under_odds)
        if under_gap <= TOTALS_MAX_DISAGREEMENT and under_ev > 0 and under_edge >= EV_THRESHOLD:
            picks.append({
                **base,
                "pick": "Under",
                "pick_side": "Under",
                "model_prob": round(blended_under, 4),
                "raw_model_prob": round(model_under, 4),
                "implied_prob": round(under_novig, 4),
                "ev": under_ev,
                "edge": under_edge,
                "units": size_bet(blended_under, under_odds),
                "odds": under_odds,
            })

    picks.sort(key=lambda x: x["ev"], reverse=True)
    logger.info(
        "Found %d +EV totals picks from %d games (sigma %.2f, blend weight %.2f).",
        len(picks), len(games_with_odds), sigma, blend_weight,
    )
    return picks


def compute_confidence(edge: float, ev: float,
                       max_disagreement: float = MAX_RAW_DISAGREEMENT,
                       weight: float | None = None) -> int:
    """Return a 1–5 star rating from where the edge sits in the achievable band.

    Because every pick's probability is blended toward the market, the maximum
    achievable edge is (1 - blend_weight) * max_disagreement — e.g. with a 0.5
    weight and the 0.15 moneyline cap, no pick can ever exceed a 7.5% edge.
    Fixed thresholds like "5 stars at 10% edge" were therefore unreachable and
    every pick clustered at 2–3 stars. Instead, the band between EV_THRESHOLD
    (the minimum edge to bet at all) and that achievable maximum is split into
    five equal tiers, so the stars adapt automatically when the self-tuned
    blend weight changes.

    Args:
        edge: The pick's blended edge over the no-vig market.
        ev: The pick's expected value (kept for API compatibility / future use).
        max_disagreement: The raw-disagreement cap for this market type
            (MAX_RAW_DISAGREEMENT for moneyline, TOTALS_MAX_DISAGREEMENT for O/U).
        weight: Blend weight used for this market; None = self-tuned moneyline weight.
    """
    if weight is None:
        weight = calibration.get_blend_weight()
    max_edge = (1.0 - weight) * max_disagreement
    span = max_edge - EV_THRESHOLD
    if span <= 0:
        # The achievable band has collapsed below EV_THRESHOLD — see
        # edge_band() below. Returning a flat 1 star (the old behaviour) made
        # every pick look equally weak. Rank against the achievable band itself
        # so the stars stay informative if a pick is somehow still produced.
        if max_edge <= 0:
            return 1
        frac = edge / max_edge
        return max(1, min(5, 1 + int(frac * 5)))
    frac = (edge - EV_THRESHOLD) / span
    if frac <= 0:
        return 1
    return min(5, 1 + int(frac * 5))


def moneyline_edge_status() -> dict:
    """Whether moneyline may be bet right now, and why.

    Mirrors totals_calibration.totals_edge_status(). ``reason`` is one of:

      no_calibration  not enough graded games to have tested the classifier
      no_edge         tested, and blending it cannot beat the pure market
      unreachable     the model is fine but the edge gate is unsatisfiable at
                      the current blend weight (see edge_band)
      ok              tested, and the model adds information
    """
    state = calibration.get_blend_state() or {}
    band = edge_band()

    if not calibration.is_self_tuned() or not state:
        n = state.get("n_games", 0)
        return {
            "bettable": not REQUIRE_MEASURED_EDGE,
            "reason": "no_calibration",
            "gate_enforced": REQUIRE_MEASURED_EDGE,
            "n_games": n,
            "n_required": calibration.MIN_CALIBRATION_GAMES,
            "edge_band": band,
            "detail": (
                f"{n} of {calibration.MIN_CALIBRATION_GAMES} graded games needed "
                f"before the classifier can be tested against the market."
            ),
        }

    if not state.get("model_adds_value", False):
        return {
            "bettable": not REQUIRE_MEASURED_EDGE,
            "reason": "no_edge",
            "gate_enforced": REQUIRE_MEASURED_EDGE,
            "n_games": state.get("n_games", 0),
            "n_required": calibration.MIN_CALIBRATION_GAMES,
            "edge_band": band,
            "detail": (
                f"Best blended log loss {state.get('log_loss')} vs "
                f"{state.get('pure_market_log_loss')} pure market on "
                f"{state.get('n_games')} graded games — the classifier is "
                f"subtracting information, not adding it."
            ),
        }

    if not band["reachable"]:
        return {
            "bettable": False,
            "reason": "unreachable",
            "gate_enforced": REQUIRE_MEASURED_EDGE,
            "n_games": state.get("n_games", 0),
            "n_required": calibration.MIN_CALIBRATION_GAMES,
            "edge_band": band,
            "detail": (
                f"At blend weight {band['weight']} no pick can exceed a "
                f"{band['max_achievable_edge']:.2%} edge, against the "
                f"{band['required_edge']:.2%} required to bet."
            ),
        }

    return {
        "bettable": True,
        "reason": "ok",
        "gate_enforced": REQUIRE_MEASURED_EDGE,
        "n_games": state.get("n_games", 0),
        "n_required": calibration.MIN_CALIBRATION_GAMES,
        "edge_band": band,
        "detail": (
            f"Blended log loss {state.get('log_loss')} beats "
            f"{state.get('pure_market_log_loss')} pure market on "
            f"{state.get('n_games')} graded games."
        ),
    }


def edge_band(max_disagreement: float = MAX_RAW_DISAGREEMENT,
              weight: float | None = None) -> dict:
    """Report whether a pick can mathematically clear EV_THRESHOLD right now.

    Blending caps a pick's edge at ``(1 - weight) * max_disagreement``. When the
    self-tuned weight rises above ``1 - EV_THRESHOLD / max_disagreement`` that
    ceiling drops below EV_THRESHOLD and the filter becomes unsatisfiable: the
    pipeline emits zero picks for that market every single day, with no error
    and no warning. At the moneyline defaults (EV_THRESHOLD 0.05, cap 0.15) the
    cutoff is weight > 0.667 — and the learned weight has been pinned at its
    0.95 ceiling, so no moneyline pick has been possible for weeks.

    Returns a dict with the weight, the achievable max edge, the required
    threshold, and a ``reachable`` flag.
    """
    if weight is None:
        weight = calibration.get_blend_weight()
    max_edge = (1.0 - weight) * max_disagreement
    return {
        "weight": round(weight, 4),
        "max_achievable_edge": round(max_edge, 5),
        "required_edge": EV_THRESHOLD,
        "reachable": max_edge >= EV_THRESHOLD,
    }


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
    record = f"{stats['wins']}W - {stats['losses']}L"
    if stats.get("pushes"):
        record += f" - {stats['pushes']}P"
    lines.append(f"  Record:           {record} ({stats['win_rate']:.1f}%)")
    lines.append(f"  Pending:          {stats['pending']}")
    lines.append(f"  Units Wagered:    {stats['total_units_wagered']:.1f}")
    lines.append(f"  Total Profit:     {stats['total_profit']:+.2f}u")
    lines.append(f"  ROI:              {stats['roi_pct']:+.2f}%")
    if stats["brier_score"] is not None:
        lines.append(f"  Brier Score:      {stats['brier_score']:.4f}")
    lines.append("=" * 50)
    lines.append("")
    return "\n".join(lines)
