"""Central configuration for the MLB betting model."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mlb_bets.db"
MODEL_DIR = BASE_DIR / "models"
LOG_FILE = BASE_DIR / "mlb_model.log"
CACHE_DIR = MODEL_DIR / "cache"
TRAINING_STATE_PATH = MODEL_DIR / "training_state.json"

# --- Retrain Scheduler ---
RETRAIN_SCHEDULE_DAY = "mon"
RETRAIN_SCHEDULE_HOUR = 6
RETRAIN_SCHEDULE_MINUTE = 0

# --- API Keys ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# --- Discord ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# --- Model Training ---
TRAINING_SEASONS = [2023, 2024, 2025]

# --- EV & Bet Sizing ---
EV_THRESHOLD  = 0.02  # Minimum edge to place a bet (2%)
KELLY_SCALE   = 40    # Multiplier: translates half-Kelly fraction → intuitive units
MIN_BET_UNITS = 0.5   # Floor: any qualifying pick bets at least this many units
MAX_BET_UNITS = 3.0   # Cap: never risk more than this per pick

# --- Feature Engineering Windows ---
PITCHER_ROLLING_DAYS = 30
BULLPEN_ROLLING_DAYS = 14

# --- Scheduler (Eastern Time) ---
MORNING_RUN_HOUR = 9
MORNING_RUN_MINUTE = 0
GRADING_HOUR = 8
GRADING_MINUTE = 0

# --- Retry Settings ---
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds

# --- The Odds API ---
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "baseball_mlb"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h"

# --- MLB Stats API ---
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
