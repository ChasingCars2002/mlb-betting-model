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
# NOTE: is_home is intentionally excluded — it was constant (always 1) and
# provided no discriminative signal during training.
FEATURE_COLUMNS = [
    # Home pitcher (season + rolling)
    "home_p_xFIP_season", "home_p_SIERA_season", "home_p_K_BB_pct_season", "home_p_WHIP_season",
    "home_p_xFIP_rolling", "home_p_SIERA_rolling", "home_p_K_BB_pct_rolling", "home_p_WHIP_rolling",
    # Away pitcher (season + rolling)
    "away_p_xFIP_season", "away_p_SIERA_season", "away_p_K_BB_pct_season", "away_p_WHIP_season",
    "away_p_xFIP_rolling", "away_p_SIERA_rolling", "away_p_K_BB_pct_rolling", "away_p_WHIP_rolling",
    # Bullpens
    "home_bullpen_era", "home_bullpen_fip",
    "away_bullpen_era", "away_bullpen_fip",
    # Home team hitting vs away pitcher hand
    "home_hit_wrc_plus", "home_hit_ops",
    # Away team hitting vs home pitcher hand
    "away_hit_wrc_plus", "away_hit_ops",
    # Park factor
    "park_factor",
]


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
        home_pitcher = get_pitcher_stats(game["home_pitcher_name"])
        away_pitcher = get_pitcher_stats(game["away_pitcher_name"])

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

        features = {
            # Home pitcher
            "home_p_xFIP_season": home_pitcher["xFIP_season"],
            "home_p_SIERA_season": home_pitcher["SIERA_season"],
            "home_p_K_BB_pct_season": home_pitcher["K_BB_pct_season"],
            "home_p_WHIP_season": home_pitcher["WHIP_season"],
            "home_p_xFIP_rolling": home_pitcher["xFIP_rolling"],
            "home_p_SIERA_rolling": home_pitcher["SIERA_rolling"],
            "home_p_K_BB_pct_rolling": home_pitcher["K_BB_pct_rolling"],
            "home_p_WHIP_rolling": home_pitcher["WHIP_rolling"],
            # Away pitcher
            "away_p_xFIP_season": away_pitcher["xFIP_season"],
            "away_p_SIERA_season": away_pitcher["SIERA_season"],
            "away_p_K_BB_pct_season": away_pitcher["K_BB_pct_season"],
            "away_p_WHIP_season": away_pitcher["WHIP_season"],
            "away_p_xFIP_rolling": away_pitcher["xFIP_rolling"],
            "away_p_SIERA_rolling": away_pitcher["SIERA_rolling"],
            "away_p_K_BB_pct_rolling": away_pitcher["K_BB_pct_rolling"],
            "away_p_WHIP_rolling": away_pitcher["WHIP_rolling"],
            # Bullpens
            "home_bullpen_era": home_bullpen["bullpen_era"],
            "home_bullpen_fip": home_bullpen["bullpen_fip"],
            "away_bullpen_era": away_bullpen["bullpen_era"],
            "away_bullpen_fip": away_bullpen["bullpen_fip"],
            # Hitting vs opposing pitcher hand
            "home_hit_wrc_plus": home_hitting["wrc_plus"],
            "home_hit_ops": home_hitting["ops"],
            "away_hit_wrc_plus": away_hitting["wrc_plus"],
            "away_hit_ops": away_hitting["ops"],
            "park_factor": get_park_factor(game["home_team"]),
        }

        return features

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

        # --- Pitcher stats (cached) ---
        hp_name = game.get("home_pitcher_name", "Unknown")
        ap_name = game.get("away_pitcher_name", "Unknown")

        hp_key = (hp_name, season)
        ap_key = (ap_name, season)

        if hp_key not in pitcher_cache:
            pitcher_cache[hp_key] = get_pitcher_stats(hp_name, season=season, use_rolling=False)
        if ap_key not in pitcher_cache:
            pitcher_cache[ap_key] = get_pitcher_stats(ap_name, season=season, use_rolling=False)

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

        row = {
            "home_p_xFIP_season": home_pitcher["xFIP_season"],
            "home_p_SIERA_season": home_pitcher["SIERA_season"],
            "home_p_K_BB_pct_season": home_pitcher["K_BB_pct_season"],
            "home_p_WHIP_season": home_pitcher["WHIP_season"],
            "home_p_xFIP_rolling": home_pitcher["xFIP_rolling"],
            "home_p_SIERA_rolling": home_pitcher["SIERA_rolling"],
            "home_p_K_BB_pct_rolling": home_pitcher["K_BB_pct_rolling"],
            "home_p_WHIP_rolling": home_pitcher["WHIP_rolling"],
            "away_p_xFIP_season": away_pitcher["xFIP_season"],
            "away_p_SIERA_season": away_pitcher["SIERA_season"],
            "away_p_K_BB_pct_season": away_pitcher["K_BB_pct_season"],
            "away_p_WHIP_season": away_pitcher["WHIP_season"],
            "away_p_xFIP_rolling": away_pitcher["xFIP_rolling"],
            "away_p_SIERA_rolling": away_pitcher["SIERA_rolling"],
            "away_p_K_BB_pct_rolling": away_pitcher["K_BB_pct_rolling"],
            "away_p_WHIP_rolling": away_pitcher["WHIP_rolling"],
            "home_bullpen_era": home_bullpen["bullpen_era"],
            "home_bullpen_fip": home_bullpen["bullpen_fip"],
            "away_bullpen_era": away_bullpen["bullpen_era"],
            "away_bullpen_fip": away_bullpen["bullpen_fip"],
            "home_hit_wrc_plus": home_hitting["wrc_plus"],
            "home_hit_ops": home_hitting["ops"],
            "away_hit_wrc_plus": away_hitting["wrc_plus"],
            "away_hit_ops": away_hitting["ops"],
            "park_factor": get_park_factor(game["home_team"]),
        }

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

    return X, y
