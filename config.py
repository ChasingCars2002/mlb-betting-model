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

# --- Supabase ---
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# --- Discord ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# --- Model Training ---
TRAINING_SEASONS = [2023, 2024, 2025]

# --- EV & Bet Sizing ---
EV_THRESHOLD  = 0.05  # Minimum edge (vs no-vig market, on the blended prob) to bet
KELLY_SCALE   = 13    # Multiplier: translates half-Kelly fraction → intuitive units
MIN_BET_UNITS = 0.5   # Floor: any qualifying pick bets at least this many units
MAX_BET_UNITS = 3.0   # Cap: never risk more than this per pick

# --- Totals (Over/Under) ---
# The score model yields a point estimate for total runs. To price an Over/Under
# we treat the game total as Normal(predicted_total, TOTALS_SIGMA) and integrate
# to get P(Over)/P(Under). ~3.0 runs is a reasonable residual SD for MLB game
# totals around a model estimate; raise it to be more conservative (fewer, only
# higher-confidence O/U picks), lower it to surface more. Totals reuse the same
# EV_THRESHOLD gate as moneyline, but a looser disagreement cap: the score
# model is analytical (not the overrating classifier the 0.15 moneyline cap
# guards against), and with the 0.5 market blend a 0.15 cap would nearly
# coincide with the EV gate and surface almost nothing. 0.30 leaves a real
# betting window while still rejecting wild (>~2.5 run) model-vs-line gaps.
TOTALS_SIGMA = 3.0
TOTALS_MAX_DISAGREEMENT = 0.30

# --- Market blending & adverse-selection guards ---
# The raw model is miscalibrated/under-dispersed and systematically overrates the
# side it picks (empirically ~8 pts vs. a sharp market). We therefore (1) de-vig the
# book consensus to a true probability, (2) shrink the model toward that consensus,
# and (3) reject picks where the model disagrees with the de-vigged market by an
# implausible margin (almost always model error, not real edge).
MARKET_BLEND_WEIGHT  = 0.5   # weight on the de-vigged market consensus when blending (0=pure model, 1=pure market)
MAX_RAW_DISAGREEMENT = 0.15  # skip a side if |model_prob - no_vig_market| exceeds this

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
ODDS_MARKETS = "h2h,totals"

# --- MLB Stats API ---
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
