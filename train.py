"""One-time training script — pull historical data, engineer features, train models."""

import hashlib
import json
import logging
import sys

import pandas as pd
from datetime import date, datetime

from config import TRAINING_SEASONS, LOG_FILE, CACHE_DIR, TRAINING_STATE_PATH
from data import get_historical_game_data
from features import build_training_features, FEATURE_COLUMNS
from model import train_models

logger = logging.getLogger(__name__)


def get_feature_columns_hash() -> str:
    """Return a 16-char SHA256 of FEATURE_COLUMNS for cache invalidation."""
    content = ",".join(FEATURE_COLUMNS)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_training_state() -> dict:
    """Read training_state.json. Returns {} if missing or malformed."""
    if not TRAINING_STATE_PATH.exists():
        return {}
    try:
        with open(TRAINING_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read training_state.json: %s. Treating as first run.", e)
        return {}


def save_training_state(state: dict):
    """Write state dict to training_state.json."""
    TRAINING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    logger.info("Training state saved to %s", TRAINING_STATE_PATH)


def get_or_build_season_features(
    season: int,
    force_rebuild: bool,
    current_hash: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load season features from cache, or build from scratch if unavailable.

    Returns (X, y). Returns empty DataFrame/Series when no game data exists
    (e.g., off-season or season not yet started).

    Args:
        season: MLB season year to load or build features for.
        force_rebuild: If True, bypass any existing cache file.
        current_hash: SHA256 of FEATURE_COLUMNS (reserved for the caller's
            schema-invalidation check; not used internally).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"features_{season}.parquet"

    if not force_rebuild and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            X = df[FEATURE_COLUMNS]
            y = df["home_win"].rename("home_win")
            logger.info("Loaded cached features for %d (%d rows).", season, len(X))
            return X, y
        except Exception as e:
            logger.warning("Cache for %d is corrupt (%s) — rebuilding.", season, e)

    logger.info("Building features for %d season...", season)
    historical = get_historical_game_data([season])

    if historical.empty:
        logger.warning("No game data found for %d — skipping.", season)
        return (
            pd.DataFrame(columns=FEATURE_COLUMNS),
            pd.Series([], dtype=int, name="home_win"),
        )

    X, y = build_training_features(historical)

    cache_df = X.copy()
    cache_df["home_win"] = y.values
    cache_df.to_parquet(cache_path, index=False)
    logger.info("Cached features for %d to %s.", season, cache_path)

    return X, y


def run_incremental_retrain(force: bool = False, current_year: int | None = None):
    """Retrain models using cached features for completed seasons.

    Completed seasons (year < current_year) load from parquet cache.
    The current season always re-fetches and rebuilds features.
    Passing force=True or detecting a feature schema change triggers
    a full rebuild of all season caches.

    Args:
        force: Wipe all caches and rebuild from scratch.
        current_year: Override the current year (used in tests).
    """
    if current_year is None:
        current_year = date.today().year

    print(f"\n{'='*60}")
    run_label = "Full Rebuild" if force else "Incremental Retrain"
    print(f"  MLB BETTING MODEL — {run_label}")
    print(f"{'='*60}")

    current_hash = get_feature_columns_hash()
    state = load_training_state()

    stored_hash = state.get("feature_columns_hash", "")
    if stored_hash and stored_hash != current_hash:
        print("\n  WARNING: Feature schema changed — rebuilding all season caches.")
        logger.warning(
            "Feature hash mismatch: stored=%s current=%s. Forcing full rebuild.",
            stored_hash, current_hash,
        )
        force = True

    seasons = list(TRAINING_SEASONS)
    if current_year not in seasons:
        seasons.append(current_year)

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []
    season_stats: dict[str, dict] = {}

    for season in seasons:
        is_current = (season == current_year)
        rebuild = force or is_current
        label = "rebuilding" if rebuild else "from cache"
        print(f"\n  Season {season}: {label}...")

        X, y = get_or_build_season_features(
            season, force_rebuild=rebuild, current_hash=current_hash
        )

        if X.empty:
            print(f"  Season {season}: no data available, skipping.")
            logger.info("No data for season %d, skipping.", season)
            continue

        all_X.append(X)
        all_y.append(y)
        season_stats[str(season)] = {"rows": len(X), "cached": not rebuild}
        print(f"           {len(X)} games loaded.")

    if not all_X:
        print("\n  ERROR: No training data available across all seasons. Aborting.")
        logger.error("No training data available. Aborting retrain.")
        return

    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)

    print(f"\n  Combined: {len(X_combined)} games across {len(all_X)} seasons.")
    print(f"  Home win rate: {y_combined.mean():.3f}\n")

    train_models(X_combined, y_combined)

    new_state = {
        "last_trained": datetime.now().isoformat(timespec="seconds"),
        "feature_columns_hash": current_hash,
        "seasons": season_stats,
    }
    save_training_state(new_state)
    print("\n  Training state saved. Models ready.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(LOG_FILE)),
        ],
    )
    print(f"\nMLB Betting Model — Full Training Pipeline")
    print(f"Seasons: {TRAINING_SEASONS} + current year (auto-detected)")
    print("=" * 50)
    run_incremental_retrain(force=True)


if __name__ == "__main__":
    main()
