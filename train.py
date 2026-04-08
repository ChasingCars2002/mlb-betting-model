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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
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
        with open(TRAINING_STATE_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read training_state.json: %s. Treating as first run.", e)
        return {}


def save_training_state(state: dict):
    """Write state dict to training_state.json."""
    TRAINING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    logger.info("Training state saved to %s", TRAINING_STATE_PATH)


def main():
    print(f"\nMLB Betting Model — Training Pipeline")
    print(f"Seasons: {TRAINING_SEASONS}")
    print("=" * 50)

    # Step 1: Pull historical game data
    logger.info("Step 1: Fetching historical game data...")
    historical = get_historical_game_data(TRAINING_SEASONS)

    if historical.empty:
        logger.error("No historical data retrieved. Check your internet connection.")
        sys.exit(1)

    print(f"\nLoaded {len(historical)} games across {TRAINING_SEASONS}")
    print(f"Home win rate: {historical['home_win'].mean():.3f}\n")

    # Step 2: Engineer features
    logger.info("Step 2: Engineering features...")
    X, y = build_training_features(historical)

    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}\n")

    # Step 3: Train and compare models
    logger.info("Step 3: Training models...")
    results = train_models(X, y)

    print("Training complete! Models saved to models/ directory.")
    print("You can now run: python main.py --run-now")


if __name__ == "__main__":
    main()
