"""Shared fixtures for the MLB betting model test suite."""

import sys
import os

import pytest

# Ensure the project root is on the path so tests can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _default_blend_weight(monkeypatch):
    """Pin the market blend weight to the static default.

    The weight is self-tuned from graded outcomes (models/blend_state.json),
    which would make pick-filtering assertions depend on whatever state file
    happens to be checked out. Tests that exercise the tuning itself override
    this explicitly.
    """
    import calibration
    from config import MARKET_BLEND_WEIGHT
    monkeypatch.setattr(calibration, "get_blend_weight", lambda: MARKET_BLEND_WEIGHT)
    monkeypatch.setattr(calibration, "is_self_tuned", lambda: False)


@pytest.fixture(autouse=True)
def _open_measured_edge_gates(monkeypatch):
    """Open both market gates so filter tests exercise the filter.

    filter_positive_ev/filter_totals_ev refuse to emit anything for a market
    that has not been measured beating the market (config.REQUIRE_MEASURED_EDGE).
    That is a separate decision from whether the EV arithmetic is right, and it
    reads its answer from checked-in state files — so leaving it live here would
    make every pick-filtering assertion depend on whichever blend_state.json
    happens to be on disk. Tests that exercise the gates override this.
    """
    import evaluate
    import totals_calibration

    monkeypatch.setattr(evaluate, "moneyline_edge_status", lambda: {
        "bettable": True, "reason": "ok", "gate_enforced": True,
        "n_games": 999, "n_required": 150, "detail": "test fixture",
    })
    monkeypatch.setattr(totals_calibration, "totals_edge_status", lambda: {
        "bettable": True, "reason": "ok", "gate_enforced": True,
        "n_games": 999, "n_required": 200, "detail": "test fixture",
    })


@pytest.fixture(autouse=True)
def _default_totals_calibration(monkeypatch):
    """Pin the learned totals numbers to their cold-start defaults.

    Same reasoning as _default_blend_weight: sigma, the level correction, and
    the totals blend weight are all fit from graded data and persisted to
    models/totals_state.json. Tests that exercise the fitting override this.
    """
    import totals_calibration
    from config import TOTALS_SIGMA, MARKET_BLEND_WEIGHT

    monkeypatch.setattr(totals_calibration, "get_totals_sigma", lambda: TOTALS_SIGMA)
    monkeypatch.setattr(totals_calibration, "get_level_adjust", lambda: 0.0)
    monkeypatch.setattr(totals_calibration, "get_totals_blend_weight",
                        lambda: MARKET_BLEND_WEIGHT)
    monkeypatch.setattr(totals_calibration, "is_self_tuned", lambda: False)
