"""Tests for data.py — name matching, safe_float, retry decorator, and team mapping."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import time

from data import (
    _safe_float,
    _match_pitcher_row,
    _to_fg_team,
    retry_on_failure,
    _default_pitcher_stats,
    _default_rolling_pitcher_stats,
)


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_int(self):
        assert _safe_float(3) == 3.0

    def test_float(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_numeric_string(self):
        assert _safe_float("4.20") == pytest.approx(4.20)

    def test_percentage_string(self):
        assert _safe_float("12.5%") == pytest.approx(12.5)

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, default=99.0) == 99.0

    def test_bad_string_returns_default(self):
        assert _safe_float("N/A") == 0.0

    def test_nan_handled(self):
        import math
        result = _safe_float(float("nan"))
        assert math.isnan(result)  # float("nan") casts to nan, which is expected


# ---------------------------------------------------------------------------
# _match_pitcher_row
# ---------------------------------------------------------------------------

def _make_stats(*names):
    """Create a minimal FanGraphs-style DataFrame with a Name column."""
    return pd.DataFrame({"Name": list(names), "xFIP": [3.5] * len(names)})


class TestMatchPitcherRow:
    def test_exact_full_name_match(self):
        stats = _make_stats("Gerrit Cole", "Nathan Eovaldi", "Shane Bieber")
        row = _match_pitcher_row(stats, "Gerrit Cole")
        assert row is not None
        assert row["Name"] == "Gerrit Cole"

    def test_case_insensitive_full_match(self):
        stats = _make_stats("Gerrit Cole")
        row = _match_pitcher_row(stats, "gerrit cole")
        assert row is not None

    def test_last_name_fallback(self):
        # Full name not present, fall back to last name
        stats = _make_stats("G. Cole", "Nathan Eovaldi")
        row = _match_pitcher_row(stats, "Gerrit Cole")
        assert row is not None
        assert row["Name"] == "G. Cole"

    def test_no_match_returns_none(self):
        stats = _make_stats("Nathan Eovaldi", "Shane Bieber")
        row = _match_pitcher_row(stats, "Gerrit Cole")
        assert row is None

    def test_multiple_last_name_matches_returns_first(self):
        # Two pitchers with last name "Smith"
        stats = _make_stats("Joe Smith", "Will Smith")
        row = _match_pitcher_row(stats, "Joe Smith")
        # Full-name exact match should win over last-name ambiguity
        assert row["Name"] == "Joe Smith"

    def test_ambiguous_last_name_returns_something(self):
        stats = _make_stats("Joe Walker", "Taijuan Walker")
        # No full-name match; last-name fallback is ambiguous
        row = _match_pitcher_row(stats, "Unknown Walker")
        assert row is not None  # returns first match, warns in logs


# ---------------------------------------------------------------------------
# _to_fg_team (FanGraphs abbreviation mapping)
# ---------------------------------------------------------------------------

class TestToFGTeam:
    def test_mapped_teams(self):
        assert _to_fg_team("KC") == "KCR"
        assert _to_fg_team("SD") == "SDP"
        assert _to_fg_team("SF") == "SFG"
        assert _to_fg_team("TB") == "TBR"
        assert _to_fg_team("WSH") == "WSN"

    def test_unmapped_teams_passthrough(self):
        for team in ["NYY", "BOS", "LAD", "ATL", "HOU"]:
            assert _to_fg_team(team) == team


# ---------------------------------------------------------------------------
# retry_on_failure decorator
# ---------------------------------------------------------------------------

class TestRetryOnFailure:
    def test_succeeds_on_first_try(self):
        call_count = {"n": 0}

        @retry_on_failure
        def always_works():
            call_count["n"] += 1
            return "ok"

        result = always_works()
        assert result == "ok"
        assert call_count["n"] == 1

    def test_retries_and_succeeds(self):
        call_count = {"n": 0}

        @retry_on_failure
        def fails_twice():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("transient error")
            return "recovered"

        with patch("data.RETRY_BACKOFF_BASE", 0):  # no sleep in tests
            with patch("time.sleep"):
                result = fails_twice()

        assert result == "recovered"
        assert call_count["n"] == 3

    def test_raises_after_max_retries(self):
        @retry_on_failure
        def always_fails():
            raise RuntimeError("permanent failure")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="permanent failure"):
                always_fails()

    def test_preserves_function_name(self):
        @retry_on_failure
        def my_special_func():
            pass

        assert my_special_func.__name__ == "my_special_func"


# ---------------------------------------------------------------------------
# Default stats helpers
# ---------------------------------------------------------------------------

class TestDefaultStats:
    def test_default_pitcher_stats_keys(self):
        d = _default_pitcher_stats()
        expected = {
            "xFIP_season", "SIERA_season", "K_BB_pct_season", "WHIP_season",
            "xFIP_rolling", "SIERA_rolling", "K_BB_pct_rolling", "WHIP_rolling",
        }
        assert set(d.keys()) == expected

    def test_default_rolling_stats_keys(self):
        d = _default_rolling_pitcher_stats()
        expected = {"xFIP_rolling", "SIERA_rolling", "K_BB_pct_rolling", "WHIP_rolling"}
        assert set(d.keys()) == expected

    def test_default_values_are_floats(self):
        for val in _default_pitcher_stats().values():
            assert isinstance(val, float)
