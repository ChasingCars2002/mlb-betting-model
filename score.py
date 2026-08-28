"""Analytical score prediction — estimates expected runs per team using existing features."""

# --- League reference scales -------------------------------------------------
# IMPORTANT: the "xFIP" feature slot is not a real xFIP. data._compute_fip()
# builds it as FIP with _FIP_CONSTANT = 3.10, whose league mean is ~4.00 — NOT
# the ~4.50 of league ERA. Dividing that slot by 4.50 (as this module used to)
# multiplied every run estimate by ~0.89, pushing the projected total ~0.5 runs
# below the market line on average and making ~2/3 of all totals picks Unders.
# Every reference constant below must be on the same scale as the feature it
# normalizes.
_LEAGUE_AVG_FIP = 4.00      # mean of the FIP slot produced by data._compute_fip()
_LEAGUE_AVG_BULLPEN = 4.00  # mean of bullpen_era / bullpen_fip
_LEAGUE_AVG_OPS = 0.740     # matches data.get_team_hitting_splits' league default
_BASE_RUNS = 4.35           # league-average runs per team per game
_HOME_ADV = 1.02            # ~2% scoring bump for home team

# The starter throws roughly 55% of a modern game's innings; the bullpen covers
# the rest. Attributing 100% of run prevention to the starter (the old
# behaviour) left the projection far more volatile than the market — the model
# swung ~1.4 runs around the line where books move ~0.4. Weighting the two
# staffs by innings share, and damping the response, keeps the estimate in a
# realistic band.
_STARTER_INNINGS_SHARE = 0.55
_PITCHING_ELASTICITY = 0.70  # runs allowed move less than 1:1 with staff quality
_HITTING_ELASTICITY = 0.90   # runs scored move slightly less than 1:1 with OPS


def _damped_ratio(value: float, league_avg: float, elasticity: float) -> float:
    """Ratio of `value` to `league_avg`, shrunk toward 1.0 by `elasticity`.

    A raw ratio treats a 20%-better pitcher as suppressing 20% of runs, which
    over-extrapolates: run scoring is a team outcome that only partly reflects
    one input. Damping keeps the projection inside a plausible range.
    """
    if not league_avg:
        return 1.0
    return 1.0 + elasticity * ((value / league_avg) - 1.0)


def predict_game_scores(features: dict, level_adjust: float | None = None) -> dict:
    """Estimate expected runs for home and away teams from game features.

    Runs allowed by a team are driven by that team's starting pitcher and its
    bullpen, weighted by innings share; runs scored are driven by the batting
    team's OPS against the opposing starter's hand. The park factor applies to
    BOTH teams — they hit in the same stadium — so e.g. Coors boosts the
    visitors' scoring too.

    The reference constants above set the projection's *level*, and twice now
    hand-tuning them has moved that level past the truth and out the other side:
    an 0.89x scale error made 67% of picks Unders, and correcting it by hand
    made 92% of picks Overs. So the level is no longer trusted to the constants.
    ``level_adjust`` — runs to subtract, learned by totals_calibration from the
    residuals of the graded full slate — carries it instead, and is 0.0 until
    there is enough data to measure it. Pass it explicitly to keep this function
    pure; None reads the learned value.

    The subtraction is split evenly between the two teams: the residuals it is
    fit on are game totals, which say nothing about how a level error divides
    between the sides.

    Returns keys: predicted_home_score, predicted_away_score, predicted_total.
    """
    if level_adjust is None:
        from totals_calibration import get_level_adjust
        level_adjust = get_level_adjust()
    # The home team scores against the AWAY staff, and vice versa.
    away_starter = features.get("away_p_xFIP_season") or _LEAGUE_AVG_FIP
    home_starter = features.get("home_p_xFIP_season") or _LEAGUE_AVG_FIP
    away_pen = features.get("away_bullpen_fip") or _LEAGUE_AVG_BULLPEN
    home_pen = features.get("home_bullpen_fip") or _LEAGUE_AVG_BULLPEN
    home_ops = features.get("home_hit_ops") or _LEAGUE_AVG_OPS
    away_ops = features.get("away_hit_ops") or _LEAGUE_AVG_OPS
    park = features.get("park_factor") or 1.0

    s, p = _STARTER_INNINGS_SHARE, 1.0 - _STARTER_INNINGS_SHARE

    # Effective staff quality faced by each offense, on the FIP scale.
    away_staff = s * away_starter + p * (away_pen * _LEAGUE_AVG_FIP / _LEAGUE_AVG_BULLPEN)
    home_staff = s * home_starter + p * (home_pen * _LEAGUE_AVG_FIP / _LEAGUE_AVG_BULLPEN)

    half_adjust = level_adjust / 2.0
    home_runs = (
        _BASE_RUNS
        * _damped_ratio(away_staff, _LEAGUE_AVG_FIP, _PITCHING_ELASTICITY)
        * _damped_ratio(home_ops, _LEAGUE_AVG_OPS, _HITTING_ELASTICITY)
        * park
        * _HOME_ADV
    ) - half_adjust
    away_runs = (
        _BASE_RUNS
        * _damped_ratio(home_staff, _LEAGUE_AVG_FIP, _PITCHING_ELASTICITY)
        * _damped_ratio(away_ops, _LEAGUE_AVG_OPS, _HITTING_ELASTICITY)
        * park
    ) - half_adjust

    # A team cannot score a negative number of runs; a large downward correction
    # on an already-low projection must not produce one.
    home_runs = round(max(0.0, home_runs), 2)
    away_runs = round(max(0.0, away_runs), 2)

    return {
        "predicted_home_score": home_runs,
        "predicted_away_score": away_runs,
        "predicted_total": round(home_runs + away_runs, 2),
    }
