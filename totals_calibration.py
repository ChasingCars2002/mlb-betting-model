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
  has_edge      whether the model beat the market on games no parameter was fit on

``has_edge`` is the one that matters, and it is deliberately hard to pass.

It is scored only on a holdout: the parameters come from the earlier window, the
score comes from the later untouched one. Scoring on the fit sample would tilt
the comparison — removing that sample's observed bias improves its RMSE by
construction, while the line, which gets no fitted parameter, gains nothing. On
a near-tie that alone would open betting on the fit's own optimism.

It also requires two things, not one. A totals bet is a claim that the
projection is closer to the truth than the line is (RMSE), *and* that the
Over/Under probabilities it implies beat the book's prices (log loss). Those can
disagree: a projection can sit nearer the realized total on average while
pricing O/U worse than the market, and it is the probabilities that the wager
is actually placed on.

When either fails, evaluate.filter_totals_ev stops emitting picks (see
config.REQUIRE_MEASURED_EDGE). No amount of blending, sigma tuning, or level
correction makes a market-losing model +EV — it only changes how fast the vig
gets paid.

One trap worth naming, because it is invisible until it has been running a
while: once a level correction is live, score.predict_game_scores subtracts it
before the projection is ever logged. A refit that reads those rows as raw would
re-measure a bias it has already corrected and subtract it twice, and with
corrected and uncorrected rows accumulating side by side it would chase a moving
mixture forever. Every logged row therefore records the adjustment applied to
it, and the fit adds it back first.
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

# The gate is scored only on games no parameter was fit on. The holdout is the
# most recent HOLDOUT_FRACTION of the log, but never fewer than
# MIN_HOLDOUT_GAMES: below that, an RMSE or log-loss comparison is noise, and a
# gate that opens on noise is the failure this whole module exists to prevent.
HOLDOUT_FRACTION = 0.35
MIN_HOLDOUT_GAMES = 80

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
      no_edge         tested on the holdout, and the model did not beat the
                      market on both RMSE and O/U log loss
      ok              tested on the holdout, and it beat the market on both
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
        "holdout_log_loss": state.get("holdout_log_loss"),
        "holdout_market_log_loss": state.get("holdout_market_log_loss"),
        "detail": (
            f"On {state.get('n_holdout')} held-out games: projection RMSE "
            f"{state.get('model_rmse')} vs posted line {state.get('market_rmse')} "
            f"({'pass' if state.get('rmse_beats_line') else 'fail'}); O/U log loss "
            f"{state.get('holdout_log_loss')} vs market "
            f"{state.get('holdout_market_log_loss')} "
            f"({'pass' if state.get('probs_beat_market') else 'fail'})."
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


def _split_index(dates: list[str]) -> Optional[int]:
    """First index of the holdout window, or None if no usable split exists.

    The cut lands on a date boundary so a single day's games never straddle it —
    games on one slate share weather, a schedule, and often an opponent, so
    splitting mid-day leaks train information into the holdout.
    """
    n = len(dates)
    target = n - max(MIN_HOLDOUT_GAMES, int(round(n * HOLDOUT_FRACTION)))
    if target < 1:
        return None
    # Walk back to the start of the date the target lands in, so the whole day
    # falls in the holdout.
    split = target
    while split > 0 and dates[split - 1] == dates[target]:
        split -= 1
    if split < 1 or n - split < MIN_HOLDOUT_GAMES:
        return None
    return split


def _decided_mask(actual: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Games that were not pushes.

    A total landing exactly on the line says nothing about which side was right,
    so it is dropped rather than assigned to one arbitrarily.
    """
    return actual != line


def _market_over_probs(over_odds: list[int], under_odds: list[int],
                       mask: np.ndarray) -> np.ndarray:
    from odds import devig_two_way
    return np.array([
        devig_two_way(o, u)[0]
        for o, u, keep in zip(over_odds, under_odds, mask) if keep
    ], dtype=float)


def _fit_blend_weight(centred: np.ndarray, line: np.ndarray, actual: np.ndarray,
                      over_odds: list[int], under_odds: list[int],
                      sigma: float):
    """Grid-search the blend weight minimizing O/U log loss on these games.

    Returns (weight, best_loss, pure_market_loss, at_ceiling), or
    (None, None, None, False) when too few games were decided to fit anything.
    """
    decided = _decided_mask(actual, line)
    if decided.sum() < MIN_HOLDOUT_GAMES:
        return None, None, None, False

    model_over = _over_probability(centred[decided], line[decided], sigma)
    market_over = _market_over_probs(over_odds, under_odds, decided)
    y = (actual[decided] > line[decided]).astype(float)

    grid = np.arange(BLEND_WEIGHT_MIN, BLEND_WEIGHT_MAX + 1e-9, 0.01)
    losses = [_blend_log_loss(w, model_over, market_over, y) for w in grid]
    best_idx = int(np.argmin(losses))
    weight = round(float(grid[best_idx]), 2)
    return (
        weight,
        round(losses[best_idx], 5),
        round(_blend_log_loss(1.0, model_over, market_over, y), 5),
        weight >= BLEND_WEIGHT_MAX - 1e-9,
    )


def _score_blend_weight(centred: np.ndarray, line: np.ndarray, actual: np.ndarray,
                        over_odds: list[int], under_odds: list[int],
                        sigma: float, weight: float):
    """Log loss of an already-chosen weight on these games, vs pure market.

    Returns (blended_loss, market_loss), or (None, None) when too few games were
    decided to score. Nothing is fit here — that is the point: the weight and
    the level correction both come from the train window.
    """
    decided = _decided_mask(actual, line)
    if decided.sum() < MIN_HOLDOUT_GAMES // 2:
        return None, None

    model_over = _over_probability(centred[decided], line[decided], sigma)
    market_over = _market_over_probs(over_odds, under_odds, decided)
    y = (actual[decided] > line[decided]).astype(float)
    return (
        round(_blend_log_loss(weight, model_over, market_over, y), 5),
        round(_blend_log_loss(1.0, model_over, market_over, y), 5),
    )


def update_totals_calibration() -> Optional[dict]:
    """Refit level, sigma, blend weight, and the skill test; persist the result.

    Returns the new state dict, or None when there is not enough graded
    full-slate data yet.
    """
    from database import get_graded_totals_log

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

    # Reconstruct the RAW projection. Once a level correction is live,
    # score.predict_game_scores has already subtracted it before the row was
    # logged, so `predicted_total` is a corrected number. Re-measuring the bias
    # on it and subtracting again would double-correct, and as corrected and
    # uncorrected rows accumulate side by side the fit would chase a moving
    # mixture and never settle. Adding the recorded adjustment back puts every
    # row on the same uncorrected scale.
    predicted = np.array(
        [r["predicted_total"] + (r.get("level_adjust_applied") or 0.0) for r in rows],
        dtype=float,
    )
    line = np.array([r["market_total"] for r in rows], dtype=float)
    actual = np.array([r["actual_total"] for r in rows], dtype=float)
    over_odds = [int(r["over_odds"]) for r in rows]
    under_odds = [int(r["under_odds"]) for r in rows]

    # --- Train / holdout split ----------------------------------------------
    # The gate decides whether to put money down, so it must not be scored on
    # the games its own parameters were fit on: removing a sample's observed
    # bias improves that sample's RMSE by construction, while the line — which
    # gets no fitted parameter — carries no such advantage. On a near-tie that
    # is enough to open betting on nothing but the fit's own optimism.
    #
    # So: fit on the earlier window, score the gate on the later untouched one.
    # Split on a date boundary so a single day never straddles it.
    split = _split_index([r["date"] for r in rows])
    if split is None:
        logger.info(
            "Totals calibration: %d graded games, but no date split leaves a "
            "%d-game holdout — gate stays closed until it does.",
            n, MIN_HOLDOUT_GAMES,
        )
        return None
    train = slice(0, split)
    test = slice(split, n)

    # --- Level: centre the projection on realized totals ---------------------
    # Fit on train for the honest gate score below; refit on everything for the
    # value actually deployed, since more data is a better estimate of a single
    # mean and the deployed number is not what the gate is scoring.
    train_bias = float(np.mean(predicted[train] - actual[train]))
    raw_bias = float(np.mean(predicted - actual))
    level_adjust = float(np.clip(raw_bias, -MAX_LEVEL_ADJUST, MAX_LEVEL_ADJUST))
    level_clamped = abs(raw_bias) > MAX_LEVEL_ADJUST
    centred = predicted - level_adjust

    # --- Sigma: residual SD of the CENTRED projection ------------------------
    # Centred, because the level error is corrected separately; folding it into
    # sigma would both understate the correction and overstate the spread.
    residual_sd = float(np.std(centred - actual, ddof=1))
    sigma = float(max(TOTALS_SIGMA_MIN, residual_sd))

    # --- Blend weight: fit on the O/U outcome itself -------------------------
    train_sigma = float(max(
        TOTALS_SIGMA_MIN,
        np.std((predicted[train] - train_bias) - actual[train], ddof=1),
    ))
    weight, best_loss, market_loss, at_ceiling = _fit_blend_weight(
        predicted[train] - train_bias, line[train], actual[train],
        over_odds[train], under_odds[train], train_sigma,
    )
    if weight is None:
        weight, at_ceiling = MARKET_BLEND_WEIGHT, False

    # --- Skill test, on the holdout only -------------------------------------
    # Two questions, and a totals bet needs both answered yes. The line gets no
    # correction in either because it needs none — it is already the market's
    # centred estimate.
    #
    #   1. Is the projection closer to the realized total than the line?
    #   2. Do the O/U probabilities it implies beat the market's own prices?
    #
    # RMSE alone is not enough: a projection can be nearer the truth on average
    # and still price Over/Under worse than the book, in which case the wager
    # placed on those probabilities is not the thing that was validated. The
    # blend weight is fit on train and scored on the holdout, so a weight pinned
    # at the ceiling — "ignore the model" — shows up here as a loss, not as a
    # passing grade.
    test_centred = predicted[test] - train_bias
    model_rmse = float(np.sqrt(np.mean((test_centred - actual[test]) ** 2)))
    market_rmse = float(np.sqrt(np.mean((line[test] - actual[test]) ** 2)))
    rmse_beats_line = model_rmse < market_rmse

    holdout_loss, holdout_market_loss = _score_blend_weight(
        test_centred, line[test], actual[test],
        over_odds[test], under_odds[test], train_sigma, weight,
    )
    probs_beat_market = (
        holdout_loss is not None
        and holdout_market_loss is not None
        and holdout_loss < holdout_market_loss
    )

    has_edge = bool(rmse_beats_line and probs_beat_market)

    state = {
        "n_games": n,
        "n_train": split,
        "n_holdout": n - split,
        "fitted": True,
        "level_adjust": round(level_adjust, 3),
        "raw_bias": round(raw_bias, 3),
        "level_clamped": level_clamped,
        "sigma": round(sigma, 3),
        "residual_sd": round(residual_sd, 3),
        # Every figure below is scored on the holdout, never on the fit sample.
        "model_rmse": round(model_rmse, 3),
        "market_rmse": round(market_rmse, 3),
        "rmse_beats_line": bool(rmse_beats_line),
        "holdout_log_loss": holdout_loss,
        "holdout_market_log_loss": holdout_market_loss,
        "probs_beat_market": bool(probs_beat_market),
        "has_edge": has_edge,
        "weight": weight,
        "log_loss": best_loss,
        "pure_market_log_loss": market_loss,
        "weight_at_ceiling": at_ceiling,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_state(state)

    if not has_edge:
        logger.error(
            "TOTALS HEALTH: the run projection failed the holdout test on %d "
            "games (RMSE %.3f vs %.3f posted line, %s; O/U log loss %s vs %s "
            "market, %s). Totals betting is a claim that both hold, so picks "
            "are suppressed until they do.",
            n - split, model_rmse, market_rmse,
            "pass" if rmse_beats_line else "FAIL",
            holdout_loss, holdout_market_loss,
            "pass" if probs_beat_market else "FAIL",
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
        "Totals calibration on %d graded games (%d train / %d holdout): level "
        "%+.2f runs, sigma %.2f, blend weight %.2f. Holdout RMSE %.3f vs line "
        "%.3f, O/U log loss %s vs %s market (edge: %s).",
        n, split, n - split, level_adjust, sigma, weight,
        model_rmse, market_rmse, holdout_loss, holdout_market_loss, has_edge,
    )
    return state


def _write_state(state: dict):
    TOTALS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOTALS_STATE_PATH.write_text(json.dumps(state, indent=2))
    _invalidate_cache()
