"""Feature engineering — build sabermetric feature vectors for model input."""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from data import (
    get_pitcher_stats,
    get_bullpen_stats,
    get_team_hitting_splits,
    _get_pitcher_hand,
    get_park_factor,
)

logger = logging.getLogger(__name__)

# Feature column order (must match training and prediction).
#
# NOTE: is_home is intentionally excluded — it was constant (always 1) and
# provided no discriminative signal during training.
#
# The eight *_rolling columns were REMOVED because they carried a train/serve
# skew that silently destroyed the model. data.get_pitcher_stats() is called
# with use_rolling=False on the training path, which copies each season value
# straight into its rolling slot — so during training the rolling columns were
# byte-identical duplicates of the season columns. At prediction time
# use_rolling=True fetches a real 30-day window, so those same eight inputs
# suddenly carried different values than anything the model saw while fitting.
# The trees had split arbitrarily across the duplicate pairs (feature
# importance was near-uniform at ~1/25 per column), and every one of those
# splits was evaluated on a shifted distribution in production.
#
# The two *_wrc_plus columns were removed because they are perfectly collinear
# with their OPS counterparts: data.get_team_hitting_splits() computes
# wrc_plus = 100 * ops / 0.720, an exact affine transform, so they added no
# information and only diluted split selection.
#
# Restoring genuine recent-form signal requires point-in-time rolling windows
# on the TRAINING path too (see docs/model-review.md); until that exists,
# season-to-date stats are the only inputs that mean the same thing in both
# training and production.
FEATURE_COLUMNS = [
    # Home pitcher (season to date)
    "home_p_xFIP_season", "home_p_SIERA_season", "home_p_K_BB_pct_season", "home_p_WHIP_season",
    # Away pitcher (season to date)
    "away_p_xFIP_season", "away_p_SIERA_season", "away_p_K_BB_pct_season", "away_p_WHIP_season",
    # Bullpens
    "home_bullpen_era", "home_bullpen_fip",
    "away_bullpen_era", "away_bullpen_fip",
    # Home team hitting vs away pitcher hand
    "home_hit_ops",
    # Away team hitting vs home pitcher hand
    "away_hit_ops",
    # Park factor
    "park_factor",
]


def _assemble_row(home_pitcher: dict, away_pitcher: dict,
                  home_bullpen: dict, away_bullpen: dict,
                  home_hitting: dict, away_hitting: dict,
                  park_factor: float) -> dict:
    """Build one FEATURE_COLUMNS-shaped row from the raw stat dicts.

    Shared by the prediction and training paths so the two can never drift
    apart in column set or ordering.
    """
    return {
        "home_p_xFIP_season": home_pitcher["xFIP_season"],
        "home_p_SIERA_season": home_pitcher["SIERA_season"],
        "home_p_K_BB_pct_season": home_pitcher["K_BB_pct_season"],
        "home_p_WHIP_season": home_pitcher["WHIP_season"],
        "away_p_xFIP_season": away_pitcher["xFIP_season"],
        "away_p_SIERA_season": away_pitcher["SIERA_season"],
        "away_p_K_BB_pct_season": away_pitcher["K_BB_pct_season"],
        "away_p_WHIP_season": away_pitcher["WHIP_season"],
        "home_bullpen_era": home_bullpen["bullpen_era"],
        "home_bullpen_fip": home_bullpen["bullpen_fip"],
        "away_bullpen_era": away_bullpen["bullpen_era"],
        "away_bullpen_fip": away_bullpen["bullpen_fip"],
        "home_hit_ops": home_hitting["ops"],
        "away_hit_ops": away_hitting["ops"],
        "park_factor": park_factor,
    }


def build_game_features(game: dict) -> Optional[dict]:
    """Build a feature vector for a single game matchup.

    Args:
        game: Dict with keys from get_todays_games() — must include
              home/away team, pitcher names, and pitcher hands.

    Returns:
        Dict of feature name → value, or None if critical data is missing.
    """
    try:
        # --- Starting pitcher stats ---
        home_pitcher = get_pitcher_stats(game.get("home_pitcher_id"), game["home_pitcher_name"])
        away_pitcher = get_pitcher_stats(game.get("away_pitcher_id"), game["away_pitcher_name"])

        # --- Bullpen stats ---
        home_bullpen = get_bullpen_stats(game["home_team"])
        away_bullpen = get_bullpen_stats(game["away_team"])

        # --- Hitting splits (matched to opposing pitcher hand) ---
        home_hitting = get_team_hitting_splits(
            game["home_team"], game["away_pitcher_hand"]
        )
        away_hitting = get_team_hitting_splits(
            game["away_team"], game["home_pitcher_hand"]
        )

        return _assemble_row(
            home_pitcher, away_pitcher, home_bullpen, away_bullpen,
            home_hitting, away_hitting, get_park_factor(game["home_team"]),
        )

    except Exception as e:
        logger.error(
            "Failed to build features for %s @ %s: %s",
            game.get("away_team", "?"), game.get("home_team", "?"), e,
        )
        return None


def build_training_features(historical_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix from historical game data for model training.

    This is a batch operation that processes all games. For games where
    pitcher stats are unavailable, league-average defaults are used.

    Args:
        historical_games: DataFrame from data.get_historical_game_data().

    Returns:
        (X, y) — feature DataFrame and target Series (1 = home win).
    """
    logger.info("Building training features for %d games...", len(historical_games))

    feature_rows = []
    targets = []

    # Cache stats to avoid repeated API calls during training
    pitcher_cache = {}
    bullpen_cache = {}
    hitting_cache = {}
    hand_cache = {}

    for idx, game in historical_games.iterrows():
        if idx % 500 == 0:
            logger.info("Processing game %d / %d...", idx, len(historical_games))

        season = game.get("season", 2024)

        # --- Pitcher hands (use actual hands from MLB API, cached) ---
        hp_id = game.get("home_pitcher_id")
        ap_id = game.get("away_pitcher_id")

        try:
            if hp_id and hp_id == hp_id and hp_id not in hand_cache:  # NaN check: NaN != NaN
                hand_cache[hp_id] = _get_pitcher_hand(int(hp_id))
            if ap_id and ap_id == ap_id and ap_id not in hand_cache:
                hand_cache[ap_id] = _get_pitcher_hand(int(ap_id))
        except (ValueError, TypeError):
            pass  # NaN or non-numeric ID — default to R below

        home_pitcher_hand = hand_cache.get(hp_id, "R")
        away_pitcher_hand = hand_cache.get(ap_id, "R")

        # --- Pitcher stats (cached by id, looked up in the MLB leaderboard) ---
        hp_name = game.get("home_pitcher_name", "Unknown")
        ap_name = game.get("away_pitcher_name", "Unknown")

        hp_key = (hp_id, season)
        ap_key = (ap_id, season)

        if hp_key not in pitcher_cache:
            pitcher_cache[hp_key] = get_pitcher_stats(hp_id, hp_name, season=season, use_rolling=False)
        if ap_key not in pitcher_cache:
            pitcher_cache[ap_key] = get_pitcher_stats(ap_id, ap_name, season=season, use_rolling=False)

        home_pitcher = pitcher_cache[hp_key]
        away_pitcher = pitcher_cache[ap_key]

        # --- Bullpen stats (cached) ---
        hb_key = (game["home_team"], season)
        ab_key = (game["away_team"], season)

        if hb_key not in bullpen_cache:
            bullpen_cache[hb_key] = get_bullpen_stats(game["home_team"], season=season)
        if ab_key not in bullpen_cache:
            bullpen_cache[ab_key] = get_bullpen_stats(game["away_team"], season=season)

        home_bullpen = bullpen_cache[hb_key]
        away_bullpen = bullpen_cache[ab_key]

        # --- Hitting splits (cached by actual pitcher hand) ---
        hh_key = (game["home_team"], away_pitcher_hand, season)
        ah_key = (game["away_team"], home_pitcher_hand, season)

        if hh_key not in hitting_cache:
            hitting_cache[hh_key] = get_team_hitting_splits(
                game["home_team"], away_pitcher_hand, season=season
            )
        if ah_key not in hitting_cache:
            hitting_cache[ah_key] = get_team_hitting_splits(
                game["away_team"], home_pitcher_hand, season=season
            )

        home_hitting = hitting_cache[hh_key]
        away_hitting = hitting_cache[ah_key]

        row = _assemble_row(
            home_pitcher, away_pitcher, home_bullpen, away_bullpen,
            home_hitting, away_hitting, get_park_factor(game["home_team"]),
        )

        feature_rows.append(row)
        targets.append(game["home_win"])

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    y = pd.Series(targets, name="home_win")

    # Fill any remaining NaN with column medians
    X = X.fillna(X.median())

    logger.info(
        "Built feature matrix: %d games x %d features. Home win rate: %.1f%%",
        len(X), len(FEATURE_COLUMNS), y.mean() * 100,
    )
    check_feature_quality(X)

    return X, y


# ---------------------------------------------------------------------------
# Data-quality guard
# ---------------------------------------------------------------------------
#
# Every upstream fetch in data.py swallows its exception and returns a
# hard-coded league-average constant. That is the right behaviour for a single
# missing pitcher on a live slate, but during a bulk training build it means a
# broken endpoint, an expired scrape, or an unhydrated field degrades thousands
# of rows to the SAME constant with nothing louder than a debug log — and the
# resulting matrix trains a model that has no signal to find.
#
# This is not hypothetical: the medians persisted by the last training run
# (models/feature_medians.joblib) are EXACTLY the fallback constants for all 24
# non-park features (xFIP 4.20, SIERA 4.20, K-BB% 10.0, WHIP 1.30, bullpen
# 4.00/4.00, OPS 0.740, wRC+ 100.0). A median lands exactly on the fallback
# only if at least half the rows ARE the fallback.

# Feature -> the league-average constant data.py substitutes on failure.
FALLBACK_VALUES = {
    "home_p_xFIP_season": 4.20, "away_p_xFIP_season": 4.20,
    "home_p_SIERA_season": 4.20, "away_p_SIERA_season": 4.20,
    "home_p_K_BB_pct_season": 10.0, "away_p_K_BB_pct_season": 10.0,
    "home_p_WHIP_season": 1.30, "away_p_WHIP_season": 1.30,
    "home_bullpen_era": 4.00, "away_bullpen_era": 4.00,
    "home_bullpen_fip": 4.00, "away_bullpen_fip": 4.00,
    "home_hit_ops": 0.740, "away_hit_ops": 0.740,
}

# Above this share of fallback rows a feature carries essentially no signal.
MAX_FALLBACK_RATE = 0.25


def feature_fallback_rates(X: pd.DataFrame) -> dict[str, float]:
    """Fraction of rows equal to the league-average fallback, per feature."""
    if X.empty:
        return {c: 0.0 for c in FALLBACK_VALUES if c in X.columns}
    return {
        col: float((X[col] == val).mean())
        for col, val in FALLBACK_VALUES.items()
        if col in X.columns
    }


def check_feature_quality(X: pd.DataFrame,
                          max_rate: float = MAX_FALLBACK_RATE) -> dict[str, float]:
    """Log the per-feature fallback rate and flag degraded features.

    Returns the rate map so callers (train.py) can abort a retrain rather than
    overwrite a working model with one fitted on league-average constants.
    """
    rates = feature_fallback_rates(X)
    degraded = {c: r for c, r in rates.items() if r > max_rate}

    for col, rate in sorted(rates.items(), key=lambda kv: -kv[1]):
        logger.info("Feature fallback rate — %-28s %.1f%%", col, rate * 100)

    if degraded:
        logger.error(
            "DATA QUALITY: %d of %d features are league-average fallbacks in "
            "more than %.0f%% of rows: %s. The upstream stat fetch is failing; "
            "a model trained on this matrix will have no signal.",
            len(degraded), len(rates), max_rate * 100,
            ", ".join(f"{c} {r:.0%}" for c, r in sorted(degraded.items(), key=lambda kv: -kv[1])),
        )
    return rates
