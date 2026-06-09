"""Self-tuning market blend — learns MARKET_BLEND_WEIGHT from graded outcomes.

The model's raw probability is blended toward the de-vigged market before any
bet is placed (see evaluate.blend_with_market). The blend weight controls how
much we trust the market vs. our own model, and the right value can only be
known empirically. This module closes that loop:

  1. Every prediction day, log the RAW model probability and the no-vig market
     probability for EVERY game with odds (not just picks — picks-only data is
     adversely selected and would bias the fit).
  2. During grading, attach the actual outcome to each logged game.
  3. After grading, grid-search the blend weight that minimizes log loss of
     `w * market + (1 - w) * model` on all graded games, and persist it.
  4. At prediction time, evaluate.py uses the learned weight (bounded, and only
     once enough graded games exist) instead of the static default.

The result is a feedback loop where the system gets better-calibrated as its
own track record grows, without ever trusting a small sample: below
MIN_CALIBRATION_GAMES the static default applies, and the learned weight is
clamped to [BLEND_WEIGHT_MIN, BLEND_WEIGHT_MAX] so a fluky stretch can never
swing it to an extreme.
"""

import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np

from config import MODEL_DIR, MARKET_BLEND_WEIGHT

logger = logging.getLogger(__name__)

BLEND_STATE_PATH = MODEL_DIR / "blend_state.json"

# Don't deviate from the static default until this many graded games exist.
MIN_CALIBRATION_GAMES = 150

# Hard bounds on the learned weight. The lower bound keeps a healthy amount of
# market shrinkage even if the model looks great over a stretch; the upper
# bound stops just short of "pure market" so the model always has a voice.
BLEND_WEIGHT_MIN = 0.30
BLEND_WEIGHT_MAX = 0.95

# In-process cache (the daily pipeline calls get_blend_weight per game side).
_state_cache: Optional[dict] = None
_state_cache_loaded = False


def _load_state() -> Optional[dict]:
    """Read blend_state.json, caching the result for the process lifetime."""
    global _state_cache, _state_cache_loaded
    if not _state_cache_loaded:
        _state_cache_loaded = True
        try:
            _state_cache = json.loads(BLEND_STATE_PATH.read_text())
        except (FileNotFoundError, ValueError, OSError):
            _state_cache = None
    return _state_cache


def _invalidate_cache():
    global _state_cache, _state_cache_loaded
    _state_cache = None
    _state_cache_loaded = False


def get_blend_state() -> Optional[dict]:
    """Return the persisted calibration state, or None if not yet learned."""
    return _load_state()


def is_self_tuned() -> bool:
    """True when the learned blend weight (not the static default) is active."""
    state = _load_state()
    return bool(state) and state.get("n_games", 0) >= MIN_CALIBRATION_GAMES


def get_blend_weight() -> float:
    """Active market blend weight for moneyline picks.

    Returns the learned weight when it was fit on a sufficient sample,
    otherwise the static MARKET_BLEND_WEIGHT default. Always within
    [BLEND_WEIGHT_MIN, BLEND_WEIGHT_MAX].
    """
    state = _load_state()
    if not state:
        return MARKET_BLEND_WEIGHT
    if state.get("n_games", 0) < MIN_CALIBRATION_GAMES:
        return MARKET_BLEND_WEIGHT
    weight = state.get("weight")
    if not isinstance(weight, (int, float)):
        return MARKET_BLEND_WEIGHT
    return float(min(BLEND_WEIGHT_MAX, max(BLEND_WEIGHT_MIN, weight)))


def log_model_predictions(games_with_odds: list[dict]) -> int:
    """Persist raw model prob + no-vig market prob for every game with odds.

    Called from the daily prediction run AFTER odds are matched but BEFORE pick
    filtering, so the log covers the full slate (no adverse selection).
    Returns the number of games logged.
    """
    from database import save_model_log
    from odds import devig_two_way

    rows = []
    for g in games_with_odds:
        model_prob = g.get("model_prob")
        if model_prob is None or model_prob == 0.5:
            # 0.5 is the "features unavailable" fallback — not a real prediction,
            # and including it would drag the fit toward pure market.
            continue
        try:
            home_novig, _ = devig_two_way(g["home_odds"], g["away_odds"])
        except (KeyError, ZeroDivisionError):
            continue
        rows.append({
            "date": g["game_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "raw_model_prob": round(float(model_prob), 4),
            "market_prob": round(float(home_novig), 4),
            "home_odds": g.get("home_odds"),
            "away_odds": g.get("away_odds"),
            "model_name": g.get("model_name", "xgboost"),
        })

    if rows:
        save_model_log(rows)
    logger.info("Logged %d model predictions for calibration.", len(rows))
    return len(rows)


def _blend_log_loss(weight: float, model: np.ndarray, market: np.ndarray,
                    y: np.ndarray) -> float:
    """Log loss of the blended home-win probability at a given weight."""
    p = weight * market + (1.0 - weight) * model
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def update_blend_weight() -> Optional[dict]:
    """Refit the blend weight on all graded logged games and persist it.

    Grid-searches weight in [BLEND_WEIGHT_MIN, BLEND_WEIGHT_MAX] (step 0.01)
    minimizing log loss. Persists the result with diagnostics to
    blend_state.json. Returns the new state dict, or None when there isn't
    enough graded data yet.
    """
    from database import get_graded_model_log

    rows = get_graded_model_log()
    n = len(rows)
    if n < MIN_CALIBRATION_GAMES:
        logger.info(
            "Blend calibration: %d graded games (< %d needed) — keeping default weight %.2f.",
            n, MIN_CALIBRATION_GAMES, MARKET_BLEND_WEIGHT,
        )
        return None

    model = np.array([r["raw_model_prob"] for r in rows], dtype=float)
    market = np.array([r["market_prob"] for r in rows], dtype=float)
    y = np.array([r["home_win"] for r in rows], dtype=float)

    grid = np.arange(BLEND_WEIGHT_MIN, BLEND_WEIGHT_MAX + 1e-9, 0.01)
    losses = [_blend_log_loss(w, model, market, y) for w in grid]
    best_idx = int(np.argmin(losses))
    best_weight = round(float(grid[best_idx]), 2)
    best_loss = round(losses[best_idx], 5)

    market_loss = round(_blend_log_loss(1.0, model, market, y), 5)
    default_loss = round(_blend_log_loss(MARKET_BLEND_WEIGHT, model, market, y), 5)

    state = {
        "weight": best_weight,
        "n_games": n,
        "log_loss": best_loss,
        "default_weight_log_loss": default_loss,
        "pure_market_log_loss": market_loss,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    BLEND_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLEND_STATE_PATH.write_text(json.dumps(state, indent=2))
    _invalidate_cache()

    logger.info(
        "Blend calibration: learned weight %.2f on %d graded games "
        "(log loss %.5f vs %.5f at default %.2f, %.5f pure market).",
        best_weight, n, best_loss, default_loss, MARKET_BLEND_WEIGHT, market_loss,
    )
    return state
