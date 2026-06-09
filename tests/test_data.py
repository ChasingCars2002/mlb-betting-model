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


# ---------------------------------------------------------------------------
# MLB Stats API metric helpers (FanGraphs replacement)
# ---------------------------------------------------------------------------

import data as data_mod
from data import (
    _ip_to_float,
    _compute_fip,
    _pitcher_metrics,
    get_pitcher_stats,
    get_bullpen_stats,
    get_team_hitting_splits,
)


@pytest.fixture(autouse=True)
def _clear_mlb_caches():
    """Reset the module-level MLB caches so tests don't leak into each other."""
    data_mod._pitching_leaderboard_cache.clear()
    data_mod._team_pitching_cache.clear()
    data_mod._team_hitting_cache.clear()
    data_mod._team_id_cache.clear()
    yield


class TestIpToFloat:
    def test_thirds(self):
        assert _ip_to_float("12.1") == pytest.approx(12 + 1 / 3)
        assert _ip_to_float("12.2") == pytest.approx(12 + 2 / 3)

    def test_whole(self):
        assert _ip_to_float("6.0") == 6.0
        assert _ip_to_float("6") == 6.0

    def test_bad_input(self):
        assert _ip_to_float(None) == 0.0
        assert _ip_to_float("nonsense") == 0.0


class TestComputeFip:
    def test_known_line(self):
        # 180 IP, 15 HR, 40 BB, 5 HBP, 200 K
        stat = {"inningsPitched": "180.0", "homeRuns": 15,
                "baseOnBalls": 40, "hitByPitch": 5, "strikeOuts": 200}
        # (13*15 + 3*45 - 2*200)/180 + 3.10 = (195+135-400)/180 + 3.10
        expected = (195 + 135 - 400) / 180 + 3.10
        assert _compute_fip(stat) == pytest.approx(round(expected, 3))

    def test_zero_innings_defaults(self):
        assert _compute_fip({"inningsPitched": "0.0"}) == 4.20


class TestPitcherMetrics:
    def test_mapping(self):
        stat = {"inningsPitched": "100.0", "homeRuns": 10, "baseOnBalls": 30,
                "hitByPitch": 3, "strikeOuts": 120, "battersFaced": 400,
                "era": "3.50", "whip": "1.10"}
        m = _pitcher_metrics(stat)
        assert m["SIERA"] == pytest.approx(3.50)        # ERA in SIERA slot
        assert m["WHIP"] == pytest.approx(1.10)
        assert m["K_BB_pct"] == pytest.approx((120 - 30) / 400 * 100)
        assert m["xFIP"] == _compute_fip(stat)          # FIP in xFIP slot


class TestGetPitcherStatsMLB:
    def _board_response(self):
        return {"stats": [{"splits": [
            {"player": {"id": 543037},
             "stat": {"inningsPitched": "100.0", "homeRuns": 10, "baseOnBalls": 30,
                      "hitByPitch": 3, "strikeOuts": 120, "battersFaced": 400,
                      "era": "3.50", "whip": "1.10"}},
        ]}]}

    def test_known_pitcher(self):
        with patch.object(data_mod, "_mlb_api_get", return_value=self._board_response()):
            s = get_pitcher_stats(543037, "Gerrit Cole", season=2026, use_rolling=False)
        assert s["WHIP_season"] == pytest.approx(1.10)
        assert s["SIERA_season"] == pytest.approx(3.50)
        # rolling mirrors season when use_rolling=False
        assert s["WHIP_rolling"] == s["WHIP_season"]

    def test_unknown_pitcher_defaults(self):
        with patch.object(data_mod, "_mlb_api_get", return_value=self._board_response()):
            s = get_pitcher_stats(999999, "Nobody", season=2026, use_rolling=False)
        assert s == _default_pitcher_stats()

    def test_none_id_defaults_without_call(self):
        with patch.object(data_mod, "_mlb_api_get", side_effect=AssertionError("no call")):
            assert get_pitcher_stats(None, "x", season=2026) == _default_pitcher_stats()

    def test_nan_id_defaults(self):
        assert get_pitcher_stats(float("nan"), "x", season=2026) == _default_pitcher_stats()


class TestGetBullpenStatsMLB:
    def test_team_pitching(self):
        responses = {
            "teams": {"teams": [{"abbreviation": "NYY", "id": 147}]},
            "teams/147/stats": {"stats": [{"splits": [
                {"stat": {"era": "3.80", "inningsPitched": "500.0", "homeRuns": 50,
                          "baseOnBalls": 150, "hitByPitch": 20, "strikeOuts": 520}}]}]},
        }
        def fake(endpoint, params=None):
            return responses[endpoint]
        with patch.object(data_mod, "_mlb_api_get", side_effect=fake):
            b = get_bullpen_stats("NYY", season=2026)
        assert b["bullpen_era"] == pytest.approx(3.80)
        assert b["bullpen_fip"] > 0

    def test_unknown_team_defaults(self):
        with patch.object(data_mod, "_mlb_api_get", return_value={"teams": []}):
            b = get_bullpen_stats("ZZZ", season=2026)
        assert b == {"bullpen_era": 4.00, "bullpen_fip": 4.00}


class TestGetTeamHittingSplitsMLB:
    def test_platoon_split(self):
        responses = {
            "teams": {"teams": [{"abbreviation": "BOS", "id": 111}]},
            "teams/111/stats": {"stats": [{"splits": [
                {"stat": {"ops": "0.800"}}]}]},
        }
        def fake(endpoint, params=None):
            return responses[endpoint]
        with patch.object(data_mod, "_mlb_api_get", side_effect=fake):
            h = get_team_hitting_splits("BOS", "R", season=2026)
        assert h["ops"] == pytest.approx(0.800)
        assert h["wrc_plus"] == pytest.approx(round(100.0 * 0.800 / 0.720, 1))

    def test_default_when_no_team(self):
        with patch.object(data_mod, "_mlb_api_get", return_value={"teams": []}):
            h = get_team_hitting_splits("ZZZ", "L", season=2026)
        assert h == {"wrc_plus": 100.0, "ops": 0.740}
