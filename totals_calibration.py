"""Self-tuning totals model — fits level, sigma, and blend weight from residuals.

The moneyline side has had a feedback loop since April (see calibration.py): log
the raw model probability against the market for the full slate, grade it, and
refit how much to trust the model. The totals side never had one. Its two most
important numbers were hand-picked constants:

  - ``_BASE_RUNS`` and friends in score.py set the *level* of the projection.
  - ``TOTALS_SIGMA`` set how confident to be about it.

Neither was ever checked against a realized total, and both were wrong. The
level was ~0.5 runs low until 2026-08-16, which made ~2/3 of all picks Unders;
the constants were then realigned by hand and the level overshot the other way,
which made 92% of picks Overs and lost 19% ROI in a fortnight. Sigma was 3.0
against a true residual SD near 4.9, which inflated every edge and turned a
model with no measurable skill into 4-8 bets a day.

Both failures have the same cause: a quantity that is *measurable* was being
guessed. This module measures them instead.

Fit on the graded full-slate log (never on the picks — that sample is selected
on the model's own disagreement with the market, which is precisely the bias
being estimated), it learns:

  level_adjust  runs to subtract from every projection to centre it on reality
  sigma         residual SD of the centred projection, floored at TOTALS_SIGMA_MIN
  weight        market blend weight minimizing log loss of the O/U probability
  has_edge      whether the projection beats the posted line at predicting the
                realized total, out of sample

``has_edge`` is the one that matters. A totals bet is a claim that the
projection is closer to the truth than the line is. When it is not, no amount of
blending, sigma tuning, or level correction makes the bet +EV — it only changes
how fast the vig is paid. So when the check fails, evaluate.filter_totals_ev
stops emitting picks (see config.REQUIRE_MEASURED_EDGE) rather than betting a
model that has been measured losing.
"""

import json
import logging
import math
from datetime import datetime
from typing import Optional

import numpy as np

from config import (
    MODEL_DIR,
    MARKET_BLEND_WEIGHT,
    TOTALS_SIGMA,
    TOTALS_SIGMA_MIN,
    REQUIRE_MEASURED_EDGE,
)

logger = logging.getLogger(__name__)

TOTALS_STATE_PATH = MODEL_DIR / "totals_state.json"

# Below this many graded full-slate games there is no fit, no correction, and
# (with REQUIRE_MEASURED_EDGE) no betting. 200 games is roughly three weeks of
# a full slate — enough that a 0.3-run level error is distinguishable from
# noise, given a ~4.5 run per-game residual SD.
MIN_TOTALS_CALIBRATION_GAMES = 200

# Same rationale as the moneyline bounds: keep real market shrinkage even on a
# hot stretch, and stop just short of "ignore the model entirely" so the ceiling
# is detectable (weight_at_ceiling) rather than silently absorbing everything.
BLEND_WEIGHT_MIN = 0.30
BLEND_WEIGHT_MAX = 0.95

# A level correction larger than this means something structural is broken in
# score.py (wrong reference scale, a dead feature), not that the projection
# needs nudging. Correcting it silently would paper over the real defect, so the
# fit clamps here and says so.
MAX_LEVEL_ADJUST = 2.0

_state_cache: Optional[dict] = None
_state_cache_loaded = False


def _load_state() -> Optional[dict]:
    """Read totals_state.json, caching for the process lifetime."""
    global _state_cache, _state_cache_loaded
    if not _state_cache_loaded:
        _state_cache_loaded = True
        try:
            _state_cache = json.loads(TOTALS_STATE_PATH.read_text())
        except (FileNotFoundError, ValueError, OSError):
            _state_cache = None
    return _state_cache


def _invalidate_cache():
    global _state_cache, _state_cache_loaded
    _state_cache = None
    _state_cache_loaded = False


def _fitted_state() -> Optional[dict]:
    """The persisted state, but only when it was fit on a sufficient sample."""
    state = _load_state()
    if not state or state.get("n_games", 0) < MIN_TOTALS_CALIBRATION_GAMES:
        return None
    return state


def get_totals_state() -> Optional[dict]:
    """Return the persisted totals calibration state, or None if never fit."""
    return _load_state()


def is_self_tuned() -> bool:
    """True when a sufficient-sample fit is driving the totals numbers."""
    return _fitted_state() is not None


def get_level_adjust() -> float:
    """Runs to subtract from a raw projection to centre it on realized totals.

    0.0 until there is a fit. Applied in score.predict_game_scores so the
    corrected number is the one stored, displayed, and priced — a correction
    applied only at pricing time would leave the dashboard showing a projection
    nobody bet.
    """
    state = _fitted_state()
    if not state:
        return 0.0
    value = state.get("level_adjust")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.0
    return float(max(-MAX_LEVEL_ADJUST, min(MAX_LEVEL_ADJUST, value)))


def get_totals_sigma() -> float:
    """Residual SD to price Over/Under against, floored at TOTALS_SIGMA_MIN."""
    state = _fitted_state()
    if not state:
        return TOTALS_SIGMA
    value = state.get("sigma")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return TOTALS_SIGMA
    return float(max(TOTALS_SIGMA_MIN, value))


def get_totals_blend_weight() -> float:
    """Market blend weight for totals, learned from graded O/U outcomes.

    Until there is a fit this is the static MARKET_BLEND_WEIGHT. Note that the
    moneyline weight is deliberately NOT reused: it is fit on a gradient-boosted
    win classifier and says nothing about an analytical run projection.
    """
    state = _fitted_state()
    if not state:
        return MARKET_BLEND_WEIGHT
    weight = state.get("weight")
    if not isinstance(weight, (int, float)) or not math.isfinite(weight):
        return MARKET_BLEND_WEIGHT
    return float(min(BLEND_WEIGHT_MAX, max(BLEND_WEIGHT_MIN, weight)))


def totals_edge_status() -> dict:
    """Whether totals may be bet right now, and why.

    Returns a dict with ``bettable`` plus enough context for the console, the
    status file, and the dashboard to say something specific instead of
    rendering an empty slate. ``reason`` is one of:

      no_calibration  not enough graded full-slate games to have tested anything
      no_edge         tested, and the projection loses to the posted line
      ok              tested, and the projection beats the posted line
    """
    state = _fitted_state()
    if not state:
        logged = (_load_state() or {}).get("n_games", 0)
        return {
            "bettable": not REQUIRE_MEASURED_EDGE,
            "reason": "no_calibration",
            "gate_enforced": REQUIRE_MEASURED_EDGE,
            "n_games": logged,
            "n_required": MIN_TOTALS_CALIBRATION_GAMES,
            "detail": (
                f"{logged} of {MIN_TOTALS_CALIBRATION_GAMES} graded full-slate games "
                f"needed before the totals model can be tested against the line."
            ),
        }

    has_edge = bool(state.get("has_edge"))
    return {
        "bettable": has_edge or not REQUIRE_MEASURED_EDGE,
        "reason": "ok" if has_edge else "no_edge",
        "gate_enforced": REQUIRE_MEASURED_EDGE,
        "n_games": state.get("n_games", 0),
        "n_required": MIN_TOTALS_CALIBRATION_GAMES,
        "model_rmse": state.get("model_rmse"),
        "market_rmse": state.get("market_rmse"),
        "detail": (
            f"Projection RMSE {state.get('model_rmse')} vs posted line "
            f"{state.get('market_rmse')} on {state.get('n_games')} graded games."
        ),
    }


def can_bet_totals() -> bool:
    """Shorthand for totals_edge_status()['bettable']."""
    return bool(totals_edge_status()["bettable"])


def _over_probability(predicted: np.ndarray, line: np.ndarray,
                      sigma: float) -> np.ndarray:
    """Vectorised P(total > line) under Normal(predicted, sigma)."""
    z = (line - predicted) / sigma
    # 1 - Phi(z), via erfc to keep the far tail accurate.
    return 0.5 * np.array([math.erfc(v / math.sqrt(2.0)) for v in z])


def _blend_log_loss(weight: float, model: np.ndarray, market: np.ndarray,
                    y: np.ndarray) -> float:
    p = weight * market + (1.0 - weight) * model
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def update_totals_calibration() -> Optional[dict]:
    """Refit level, sigma, blend weight, and the skill test; persist the result.

    Returns the new state dict, or None when there is not enough graded
    full-slate data yet.
    """
    from database import get_graded_totals_log
    from odds import devig_two_way

    rows = get_graded_totals_log()
    n = len(rows)
    if n < MIN_TOTALS_CALIBRATION_GAMES:
        logger.info(
            "Totals calibration: %d graded games (< %d needed) — projections "
            "stay uncorrected and totals betting stays gated.",
            n, MIN_TOTALS_CALIBRATION_GAMES,
        )
        # Persist the count anyway so the dashboard can show progress toward
        # the threshold instead of an unexplained empty slate — but never
        # overwrite a real fit with a progress stub (the log only shrinks if the
        # database was rebuilt, and a stale fit beats none).
        if not (_load_state() or {}).get("fitted"):
            _write_state({
                "n_games": n,
                "fitted": False,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            })
        return None

    predicted = np.array([r["predicted_total"] for r in rows], dtype=float)
    line = np.array([r["market_total"] for r in rows], dtype=float)
    actual = np.array([r["actual_total"] for r in rows], dtype=float)

    # --- Level: centre the projection on realized totals ---------------------
    raw_bias = float(np.mean(predicted - actual))
    level_adjust = float(np.clip(raw_bias, -MAX_LEVEL_ADJUST, MAX_LEVEL_ADJUST))
    level_clamped = abs(raw_bias) > MAX_LEVEL_ADJUST
    centred = predicted - level_adjust

    # --- Sigma: residual SD of the CENTRED projection ------------------------
    # Centred, because the level error is corrected separately; folding it into
    # sigma would both understate the correction and overstate the spread.
    residual_sd = float(np.std(centred - actual, ddof=1))
    sigma = float(max(TOTALS_SIGMA_MIN, residual_sd))

    # --- Skill test: does the projection beat the line? ----------------------
    # RMSE against the realized total, on the same games. The line gets no
    # correction because it needs none — it is already the market's centred
    # estimate. Betting a total is a claim this comparison favours the model.
    model_rmse = float(np.sqrt(np.mean((centred - actual) ** 2)))
    market_rmse = float(np.sqrt(np.mean((line - actual) ** 2)))
    has_edge = model_rmse < market_rmse

    # --- Blend weight: fit on the O/U outcome itself -------------------------
    # Pushes (actual exactly on the line) carry no information about which side
    # was right and are dropped rather than assigned arbitrarily.
    decided = actual != line
    weight = MARKET_BLEND_WEIGHT
    best_loss = market_loss = None
    at_ceiling = False
    if decided.sum() >= MIN_TOTALS_CALIBRATION_GAMES // 2:
        model_over = _over_probability(centred[decided], line[decided], sigma)
        market_over = np.array([
            devig_two_way(int(r["over_odds"]), int(r["under_odds"]))[0]
            for r, keep in zip(rows, decided) if keep
        ], dtype=float)
        y = (actual[decided] > line[decided]).astype(float)

        grid = np.arange(BLEND_WEIGHT_MIN, BLEND_WEIGHT_MAX + 1e-9, 0.01)
        losses = [_blend_log_loss(w, model_over, market_over, y) for w in grid]
        best_idx = int(np.argmin(losses))
        weight = round(float(grid[best_idx]), 2)
        best_loss = round(losses[best_idx], 5)
        market_loss = round(_blend_log_loss(1.0, model_over, market_over, y), 5)
        at_ceiling = weight >= BLEND_WEIGHT_MAX - 1e-9

    state = {
        "n_games": n,
        "fitted": True,
        "level_adjust": round(level_adjust, 3),
        "raw_bias": round(raw_bias, 3),
        "level_clamped": level_clamped,
        "sigma": round(sigma, 3),
        "residual_sd": round(residual_sd, 3),
        "model_rmse": round(model_rmse, 3),
        "market_rmse": round(market_rmse, 3),
        "has_edge": bool(has_edge),
        "weight": weight,
        "log_loss": best_loss,
        "pure_market_log_loss": market_loss,
        "weight_at_ceiling": at_ceiling,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_state(state)

    if not has_edge:
        logger.error(
            "TOTALS HEALTH: the run projection does not beat the posted line "
            "(RMSE %.3f vs %.3f on %d graded games). Totals betting is a claim "
            "that it does, so picks are suppressed until it holds.",
            model_rmse, market_rmse, n,
        )
    if level_clamped:
        logger.error(
            "TOTALS HEALTH: raw level error is %+.2f runs, beyond the %.1f-run "
            "correction cap. That is a structural fault in score.py, not a "
            "calibration offset — fix the projection, do not lean on the clamp.",
            raw_bias, MAX_LEVEL_ADJUST,
        )
    if residual_sd < TOTALS_SIGMA_MIN:
        logger.warning(
            "Totals calibration: measured residual SD %.2f is below the %.2f "
            "floor; using the floor. A real MLB total is not that predictable.",
            residual_sd, TOTALS_SIGMA_MIN,
        )

    logger.info(
        "Totals calibration on %d graded games: level %+.2f runs, sigma %.2f, "
        "blend weight %.2f, model RMSE %.3f vs line %.3f (edge: %s).",
        n, level_adjust, sigma, weight, model_rmse, market_rmse, has_edge,
    )
    return state


def _write_state(state: dict):
    TOTALS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOTALS_STATE_PATH.write_text(json.dumps(state, indent=2))
    _invalidate_cache()
