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
