"""Tests for the totals calibration loop and the measured-edge gates.

These cover the failure that produced August 2026: a run projection whose level
and residual SD were hand-picked constants, priced with a sigma ~40% below the
truth, bet with no test of whether it beat the line, and losing 19% ROI over
126 bets while the pipeline reported nothing unusual.
"""

import json

import pytest

import calibration
import database
import evaluate
import totals_calibration
from config import TOTALS_SIGMA, TOTALS_SIGMA_MIN
from score import predict_game_scores


# conftest pins the learned totals numbers and forces both gates open, so that
# the rest of the suite tests EV arithmetic rather than whatever state files are
# checked in. The tests below are about those pinned things, so they need the
# real implementations back. Captured at import, before any fixture patches.
_REAL = {
    "get_totals_sigma": totals_calibration.get_totals_sigma,
    "get_level_adjust": totals_calibration.get_level_adjust,
    "get_totals_blend_weight": totals_calibration.get_totals_blend_weight,
    "is_self_tuned": totals_calibration.is_self_tuned,
    "totals_edge_status": totals_calibration.totals_edge_status,
}
_REAL_ML_GATE = evaluate.moneyline_edge_status


@pytest.fixture
def real_calibration(monkeypatch):
    """Undo conftest's pins for tests that exercise the calibration itself."""
    for name, fn in _REAL.items():
        monkeypatch.setattr(totals_calibration, name, fn)
    monkeypatch.setattr(evaluate, "moneyline_edge_status", _REAL_ML_GATE)


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    """Point the totals state file at a temp path and clear the cache."""
    path = tmp_path / "totals_state.json"
    monkeypatch.setattr(totals_calibration, "TOTALS_STATE_PATH", path)
    totals_calibration._invalidate_cache()
    yield path
    totals_calibration._invalidate_cache()


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db)
    database.init_db()
    return db


def _write_state(path, **fields):
    state = {"n_games": 500, "fitted": True}
    state.update(fields)
    path.write_text(json.dumps(state))
    totals_calibration._invalidate_cache()
    return state


# ---------------------------------------------------------------------------
# sigma
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("real_calibration")
class TestTotalsSigma:
    def test_falls_back_to_config_without_a_fit(self, tmp_state):
        assert totals_calibration.get_totals_sigma() == TOTALS_SIGMA

    def test_config_default_is_not_the_old_understated_three(self):
        """3.0 was the number that manufactured picks out of noise.

        The residual SD of an MLB game total around any projection is ~4.5
        runs; the market's own is ~4.6. A sigma materially below that inflates
        |P(Over) - 0.5| on every game at once, which inflates every edge past
        the 5% gate and oversizes every Kelly stake behind it.
        """
        assert TOTALS_SIGMA >= 4.0

    def test_uses_learned_sigma_once_fit(self, tmp_state):
        _write_state(tmp_state, sigma=4.8)
        assert totals_calibration.get_totals_sigma() == pytest.approx(4.8)

    def test_learned_sigma_is_floored(self, tmp_state):
        """A lucky stretch must not make the model look sharper than baseball."""
        _write_state(tmp_state, sigma=1.2)
        assert totals_calibration.get_totals_sigma() == TOTALS_SIGMA_MIN

    def test_small_sample_fit_is_ignored(self, tmp_state):
        _write_state(tmp_state, sigma=2.0,
                     n_games=totals_calibration.MIN_TOTALS_CALIBRATION_GAMES - 1)
        assert totals_calibration.get_totals_sigma() == TOTALS_SIGMA

    def test_understated_sigma_inflates_the_edge(self):
        """The mechanism itself: same projection, same line, different bet."""
        at_three = evaluate.total_over_probability(9.5, 8.5, sigma=3.0)
        at_measured = evaluate.total_over_probability(9.5, 8.5, sigma=4.9)
        # Both say Over is more likely; only the understated one clears the gate
        # after a 50/50 blend against a 0.5 market (edge = (p - 0.5) / 2).
        assert (at_three - 0.5) / 2 >= 0.05
        assert (at_measured - 0.5) / 2 < 0.05


# ---------------------------------------------------------------------------
# level correction
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("real_calibration")
class TestLevelAdjust:
    def test_zero_without_a_fit(self, tmp_state):
        assert totals_calibration.get_level_adjust() == 0.0

    def test_clamped_to_max(self, tmp_state):
        _write_state(tmp_state, level_adjust=9.0)
        assert totals_calibration.get_level_adjust() == totals_calibration.MAX_LEVEL_ADJUST

    def test_score_model_applies_it_symmetrically(self):
        base = predict_game_scores({}, level_adjust=0.0)
        shifted = predict_game_scores({}, level_adjust=1.0)
        assert shifted["predicted_total"] == pytest.approx(base["predicted_total"] - 1.0, abs=1e-2)
        # Split evenly: game-total residuals say nothing about which side is off.
        assert (base["predicted_home_score"] - shifted["predicted_home_score"]) == pytest.approx(0.5, abs=1e-2)
        assert (base["predicted_away_score"] - shifted["predicted_away_score"]) == pytest.approx(0.5, abs=1e-2)

    def test_never_projects_negative_runs(self):
        out = predict_game_scores({}, level_adjust=totals_calibration.MAX_LEVEL_ADJUST * 10)
        assert out["predicted_home_score"] >= 0
        assert out["predicted_away_score"] >= 0


# ---------------------------------------------------------------------------
# update_totals_calibration
# ---------------------------------------------------------------------------

def _seed(n, *, bias, model_noise, market_noise, seed=7):
    """Seed the log so the model's error is (bias, model_noise) and the line's
    is (0, market_noise), both against a known realized total."""
    import random
    rng = random.Random(seed)
    rows, actuals = [], []
    for i in range(n):
        actual = float(max(0, round(rng.gauss(8.6, 4.4))))
        actuals.append(actual)
        rows.append({
            "date": f"2026-07-{(i % 28) + 1:02d}",
            "home_team": f"H{i}",
            "away_team": f"A{i}",
            "raw_model_prob": 0.55,
            "market_prob": 0.52,
            "home_odds": -110,
            "away_odds": -110,
            "model_name": "xgboost",
            "predicted_total": round(actual + bias + rng.gauss(0, model_noise), 2),
            "market_total": round((actual + rng.gauss(0, market_noise)) * 2) / 2,
            "over_odds": -110,
            "under_odds": -110,
        })
    database.save_model_log(rows)
    with database.get_connection() as conn:
        for r, actual in zip(rows, actuals):
            conn.execute(
                "UPDATE model_log SET home_win = 1, actual_total = ? "
                "WHERE date = ? AND home_team = ? AND away_team = ?",
                (actual, r["date"], r["home_team"], r["away_team"]),
            )
    return rows


@pytest.mark.usefixtures("real_calibration")
class TestUpdateTotalsCalibration:
    def test_returns_none_below_threshold(self, tmp_db, tmp_state):
        _seed(20, bias=0.0, model_noise=3.0, market_noise=3.0)
        assert totals_calibration.update_totals_calibration() is None

    def test_records_progress_below_threshold(self, tmp_db, tmp_state):
        """The dashboard needs a count, not an unexplained empty slate."""
        _seed(20, bias=0.0, model_noise=3.0, market_noise=3.0)
        totals_calibration.update_totals_calibration()
        state = json.loads(tmp_state.read_text())
        assert state["fitted"] is False
        assert state["n_games"] == 20

    def test_recovers_a_known_level_bias(self, tmp_db, tmp_state):
        _seed(400, bias=1.0, model_noise=2.0, market_noise=2.0)
        state = totals_calibration.update_totals_calibration()
        assert state["level_adjust"] == pytest.approx(1.0, abs=0.3)

    def test_sigma_is_fit_on_the_centred_projection(self, tmp_db, tmp_state):
        """Sigma must measure spread, not absorb the level error too."""
        _seed(400, bias=2.0, model_noise=2.0, market_noise=2.0)
        state = totals_calibration.update_totals_calibration()
        # Residual SD around the CENTRED projection is model_noise (~2.0), not
        # the ~2.8 you would get by leaving the 2.0 bias in.
        assert state["residual_sd"] == pytest.approx(2.0, abs=0.4)

    def test_flags_edge_when_model_beats_the_line(self, tmp_db, tmp_state):
        _seed(400, bias=0.0, model_noise=1.0, market_noise=3.0)
        state = totals_calibration.update_totals_calibration()
        assert state["has_edge"] is True
        assert state["model_rmse"] < state["market_rmse"]

    def test_flags_no_edge_when_the_line_beats_the_model(self, tmp_db, tmp_state):
        """August's actual shape: projection noisier than the posted line."""
        _seed(400, bias=0.0, model_noise=4.0, market_noise=1.0)
        state = totals_calibration.update_totals_calibration()
        assert state["has_edge"] is False
        assert state["model_rmse"] > state["market_rmse"]

    def test_level_error_beyond_the_cap_is_flagged_not_absorbed(self, tmp_db, tmp_state):
        """A 3-run level error is a broken projection, not an offset to trim."""
        _seed(400, bias=3.0, model_noise=1.0, market_noise=1.0)
        state = totals_calibration.update_totals_calibration()
        assert state["level_clamped"] is True
        assert abs(state["level_adjust"]) == totals_calibration.MAX_LEVEL_ADJUST

    def test_fit_uses_the_full_slate_not_the_picks(self, tmp_db, tmp_state):
        """Regression guard: the fit must read model_log, never predictions.

        predictions only holds games the filter chose to bet, selected on the
        model's own disagreement with the line — the exact bias being measured.
        """
        _seed(400, bias=0.0, model_noise=1.0, market_noise=3.0)
        database.save_predictions([{
            "date": "2026-07-01", "home_team": "ZZZ", "away_team": "YYY",
            "pick": "Over", "pick_side": "Over", "model_prob": 0.6,
            "implied_prob": 0.5, "ev": 0.1, "edge": 0.1, "units": 1.0,
            "odds": -110, "model_name": "xgboost", "home_pitcher": "",
            "away_pitcher": "", "listed_total": 8.5, "predicted_total": 99.0,
        }], bet_type="totals")
        state = totals_calibration.update_totals_calibration()
        # A 99-run projection in `predictions` would wreck every statistic here.
        assert state["n_games"] == 400
        assert abs(state["level_adjust"]) < 1.0


# ---------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("real_calibration")
class TestTotalsGate:
    def test_closed_without_calibration(self, tmp_state):
        status = totals_calibration.totals_edge_status()
        assert status["bettable"] is False
        assert status["reason"] == "no_calibration"

    def test_closed_when_the_line_beats_the_model(self, tmp_state):
        _write_state(tmp_state, has_edge=False, model_rmse=4.9, market_rmse=4.6)
        status = totals_calibration.totals_edge_status()
        assert status["bettable"] is False
        assert status["reason"] == "no_edge"

    def test_open_when_the_model_beats_the_line(self, tmp_state):
        _write_state(tmp_state, has_edge=True, model_rmse=4.2, market_rmse=4.6)
        status = totals_calibration.totals_edge_status()
        assert status["bettable"] is True
        assert status["reason"] == "ok"

    def test_gate_can_be_disabled_by_config(self, tmp_state, monkeypatch):
        monkeypatch.setattr(totals_calibration, "REQUIRE_MEASURED_EDGE", False)
        _write_state(tmp_state, has_edge=False, model_rmse=4.9, market_rmse=4.6)
        status = totals_calibration.totals_edge_status()
        assert status["bettable"] is True
        assert status["reason"] == "no_edge"


class TestFilterHonoursTheGates:
    """The gate has to actually stop picks, not just describe the situation."""

    def _totals_game(self):
        return {
            "game_date": "2026-08-20", "home_team": "NYY", "away_team": "BOS",
            "total_line": 8.5, "over_odds": -110, "under_odds": -110,
            "predicted_total": 11.0, "model_name": "xgboost",
        }

    def _ml_game(self):
        return {
            "game_date": "2026-08-20", "home_team": "NYY", "away_team": "BOS",
            "model_prob": 0.58, "home_odds": 120, "away_odds": -140,
        }

    def test_totals_suppressed_when_gate_closed(self, monkeypatch):
        monkeypatch.setattr(totals_calibration, "totals_edge_status", lambda: {
            "bettable": False, "reason": "no_edge", "detail": "line wins",
        })
        assert evaluate.filter_totals_ev([self._totals_game()]) == []

    def test_totals_emitted_when_gate_open(self):
        # The autouse fixture opens the gate; this is the control case that
        # proves the suppression above is the gate and not a broken game dict.
        assert evaluate.filter_totals_ev([self._totals_game()])

    def test_moneyline_suppressed_when_gate_closed(self, monkeypatch):
        monkeypatch.setattr(evaluate, "moneyline_edge_status", lambda: {
            "bettable": False, "reason": "no_edge", "detail": "classifier loses",
        })
        assert evaluate.filter_positive_ev([self._ml_game()]) == []

    def test_moneyline_emitted_when_gate_open(self):
        assert evaluate.filter_positive_ev([self._ml_game()])


@pytest.mark.usefixtures("real_calibration")
class TestMoneylineGate:
    def _state(self, monkeypatch, **fields):
        state = {"n_games": 910, "weight": 0.95, "log_loss": 0.67952,
                 "pure_market_log_loss": 0.67907, "model_adds_value": False}
        state.update(fields)
        monkeypatch.setattr(calibration, "get_blend_state", lambda: state)
        monkeypatch.setattr(calibration, "is_self_tuned", lambda: True)
        monkeypatch.setattr(calibration, "get_blend_weight", lambda: state["weight"])
        return state

    def test_closed_without_calibration(self, monkeypatch):
        monkeypatch.setattr(calibration, "is_self_tuned", lambda: False)
        monkeypatch.setattr(calibration, "get_blend_state", lambda: None)
        status = evaluate.moneyline_edge_status()
        assert status["bettable"] is False
        assert status["reason"] == "no_calibration"

    def test_closed_when_the_classifier_subtracts_information(self, monkeypatch):
        """The live state: 910 graded games, blending cannot beat pure market."""
        self._state(monkeypatch, model_adds_value=False)
        status = evaluate.moneyline_edge_status()
        assert status["bettable"] is False
        assert status["reason"] == "no_edge"

    def test_reports_unreachable_separately_from_no_edge(self, monkeypatch):
        """A working model can still be gated by an unsatisfiable edge band.

        Distinguishing the two matters: 'no_edge' means retrain, 'unreachable'
        means the threshold and the blend weight are inconsistent.
        """
        self._state(monkeypatch, model_adds_value=True, weight=0.95)
        status = evaluate.moneyline_edge_status()
        assert status["bettable"] is False
        assert status["reason"] == "unreachable"

    def test_open_when_the_model_adds_value_and_the_band_is_reachable(self, monkeypatch):
        self._state(monkeypatch, model_adds_value=True, weight=0.50)
        status = evaluate.moneyline_edge_status()
        assert status["bettable"] is True
        assert status["reason"] == "ok"
