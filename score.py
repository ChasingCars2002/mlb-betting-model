"""Analytical score prediction — estimates expected runs per team using existing features."""

import math

from config import SCORE_SIGMA

_LEAGUE_AVG_ERA = 4.50
_LEAGUE_AVG_OPS = 0.720
_BASE_RUNS      = 4.5
_HOME_ADV       = 1.02  # ~2% scoring bump for home team


def predict_game_scores(features: dict) -> dict:
    """Estimate expected runs for home and away teams from game features.

    Uses away starter xFIP and home team OPS to project home runs, and vice
    versa for away runs.  Park factor is applied to the home team's offense.
    Returns keys: predicted_home_score, predicted_away_score, predicted_total.
    """
    home_xfip = features.get("away_p_xFIP_season", _LEAGUE_AVG_ERA) or _LEAGUE_AVG_ERA
    away_xfip = features.get("home_p_xFIP_season", _LEAGUE_AVG_ERA) or _LEAGUE_AVG_ERA
    home_ops  = features.get("home_hit_ops", _LEAGUE_AVG_OPS) or _LEAGUE_AVG_OPS
    away_ops  = features.get("away_hit_ops", _LEAGUE_AVG_OPS) or _LEAGUE_AVG_OPS
    park      = features.get("park_factor", 1.0) or 1.0

    home_runs = round(
        _BASE_RUNS
        * (home_xfip / _LEAGUE_AVG_ERA)
        * (home_ops  / _LEAGUE_AVG_OPS)
        * park
        * _HOME_ADV,
        2,
    )
    away_runs = round(
        _BASE_RUNS
        * (away_xfip / _LEAGUE_AVG_ERA)
        * (away_ops  / _LEAGUE_AVG_OPS),
        2,
    )

    return {
        "predicted_home_score": home_runs,
        "predicted_away_score": away_runs,
        "predicted_total": round(home_runs + away_runs, 2),
    }


def runs_delta_to_prob(predicted_total: float, total_line: float, sigma: float = None) -> tuple[float, float]:
    """Convert the model's predicted run total vs. the bookmaker line into probabilities.

    Uses a logistic (sigmoid) transform so that:
      - delta = 0   → P(over) = 0.50  (no edge)
      - delta = +σ  → P(over) ≈ 0.73  (moderate over lean)
      - delta = -σ  → P(over) ≈ 0.27  (moderate under lean)

    Args:
        predicted_total: Model's predicted total runs.
        total_line:      Bookmaker's posted total.
        sigma:           Scaling constant (defaults to SCORE_SIGMA from config).

    Returns:
        (prob_over, prob_under) — both in (0, 1), sum to 1.
    """
    if sigma is None:
        sigma = SCORE_SIGMA
    delta    = predicted_total - total_line
    prob_over = 1.0 / (1.0 + math.exp(-delta / sigma))
    return round(prob_over, 4), round(1.0 - prob_over, 4)
