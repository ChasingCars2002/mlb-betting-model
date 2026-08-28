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
# we treat the game total as Normal(predicted_total, sigma) and integrate to get
# P(Over)/P(Under).
#
# sigma is the residual SD of the actual game total around the projection. It is
# NOT a taste knob: it is a measurable property of the model, and getting it
# wrong corrupts every downstream number. Too small a sigma inflates
# |P(Over) - 0.5|, which inflates the edge, which manufactures picks that do not
# exist and oversizes the Kelly stake on all of them.
#
# The old 3.0 was a guess and it was far too low. Measured against stored final
# scores, the residual SD of `predicted_total` is ~4.9 runs and even the closing
# line's is ~4.6 — a single MLB game total is simply that noisy. 3.0 is why the
# pipeline fired 4-8 totals bets a day off a model with no demonstrated skill.
#
# The value below is only the cold-start fallback. Once enough full-slate games
# have been logged and graded, totals_calibration fits sigma from real residuals
# and that learned value is used instead (floored at TOTALS_SIGMA_MIN so a lucky
# stretch can never make the model look sharper than MLB scoring allows).
TOTALS_SIGMA = 4.3
TOTALS_SIGMA_MIN = 3.5

# Totals reuse the same EV_THRESHOLD gate as moneyline, but a looser
# disagreement cap: the score model is analytical (not the overrating classifier
# the 0.15 moneyline cap guards against), and with the 0.5 market blend a 0.15
# cap would nearly coincide with the EV gate and surface almost nothing. 0.30
# leaves a real betting window while still rejecting wild model-vs-line gaps.
TOTALS_MAX_DISAGREEMENT = 0.30

# --- Market blending & adverse-selection guards ---
# The raw model is miscalibrated/under-dispersed and systematically overrates the
# side it picks (empirically ~8 pts vs. a sharp market). We therefore (1) de-vig the
# book consensus to a true probability, (2) shrink the model toward that consensus,
# and (3) reject picks where the model disagrees with the de-vigged market by an
# implausible margin (almost always model error, not real edge).
MARKET_BLEND_WEIGHT  = 0.5   # weight on the de-vigged market consensus when blending (0=pure model, 1=pure market)
MAX_RAW_DISAGREEMENT = 0.15  # skip a side if |model_prob - no_vig_market| exceeds this

# --- Measured-edge gate -----------------------------------------------------
# Do not bet a market until the model has been shown, out of sample on the full
# logged slate, to beat the closing market at the thing the bet depends on:
# moneyline on log loss of the win probability, totals on squared error of the
# projected total. Blending toward the market caps how much damage a bad model
# can do per bet, but it cannot make a market-losing model +EV -- it just loses
# the vig more slowly. This gate is the difference between "the model has no
# edge" (fine, sit out) and "the model has no edge and bets anyway" (the
# -19% ROI August that prompted it).
#
# Setting this False resumes betting on an unvalidated model. Leave it True.
REQUIRE_MEASURED_EDGE = True

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
