"""Analytical score prediction — estimates expected runs per team using existing features."""

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
