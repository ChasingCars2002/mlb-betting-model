"""Tests for the dashboard export regression guard in main.py.

On 2026-07-03 a pipeline run in an environment with a stale checkout (its
mlb_bets.db was days old) exported dashboard JSON and pushed it, wiping the
published picks/grades. The guard refuses to overwrite picks_history.json
with data that has fewer graded entries or an older latest graded date.
"""

import json

import pytest

from main import _export_would_regress, _graded_fingerprint


def _entry(d, status):
    return {"date": d, "status": status, "pick": "X"}


FRESH = [
    _entry("2026-07-01", "Win"),
    _entry("2026-07-02", "Loss"),
    _entry("2026-07-02", "Win"),
]
STALE = [
    _entry("2026-06-29", "Win"),
    _entry("2026-06-29", "Pending"),
]


@pytest.fixture
def history_file(tmp_path):
    return tmp_path / "picks_history.json"


def test_fingerprint_counts_only_graded():
    assert _graded_fingerprint(STALE) == ("2026-06-29", 1)
    assert _graded_fingerprint(FRESH) == ("2026-07-02", 3)
    assert _graded_fingerprint([]) == ("", 0)


def test_stale_export_is_rejected(history_file):
    history_file.write_text(json.dumps(FRESH))
    assert _export_would_regress(STALE, history_file) is True


def test_fewer_graded_rows_same_date_is_rejected(history_file):
    history_file.write_text(json.dumps(FRESH))
    assert _export_would_regress(FRESH[:2], history_file) is True


def test_equal_export_is_allowed(history_file):
    history_file.write_text(json.dumps(FRESH))
    assert _export_would_regress(FRESH, history_file) is False


def test_newer_export_is_allowed(history_file):
    history_file.write_text(json.dumps(STALE))
    assert _export_would_regress(FRESH, history_file) is False


def test_missing_file_is_allowed(history_file):
    assert _export_would_regress(STALE, history_file) is False


def test_corrupt_file_is_allowed(history_file):
    history_file.write_text("{not json")
    assert _export_would_regress(STALE, history_file) is False


def test_non_list_file_is_allowed(history_file):
    history_file.write_text(json.dumps({"oops": 1}))
    assert _export_would_regress(STALE, history_file) is False


def test_force_env_override(history_file, monkeypatch):
    history_file.write_text(json.dumps(FRESH))
    monkeypatch.setenv("FORCE_DASHBOARD_EXPORT", "1")
    assert _export_would_regress(STALE, history_file) is False
