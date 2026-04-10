# Feature Engineering V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace leaky season-level FanGraphs stats with MLB Stats API rolling game logs, adding park factors, rest/fatigue, rolling team offense, and lineup quality features.

**Architecture:** Two new data-layer functions fetch and cache per-game pitcher and team batting logs per season; four pure computation functions transform those logs into feature dicts using only data before each game date; `build_training_features` and `build_game_features` call the same computation functions, eliminating train/serve skew.

**Tech Stack:** Python 3.11+, pandas, requests, pytest, MLB Stats API (`statsapi.mlb.com/api/v1`)

**Spec:** `docs/superpowers/specs/2026-04-09-feature-engineering-v2-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `data.py` | Modify | Add 5 new functions; remove 3 old ones at end |
| `features.py` | Rewrite | New FEATURE_COLUMNS, PARK_FACTORS, 4 compute functions, rewrite both build functions |
| `train.py` | Modify | `get_or_build_season_features` loads new cache files and injects into `build_training_features` |
| `tests/test_features.py` | Create | Unit tests for all compute functions and defaults |
| `tests/test_data.py` | Modify | Add tests for new data functions |

---

## MLB Stats API Reference

Boxscore endpoint: `GET /api/v1/game/{game_id}/boxscore`

```
teams.home.pitchers              → [player_id, ...]  first = starter
teams.home.players.ID{pid}
  .person.id / .fullName
  .stats.pitching
    .inningsPitched              "6.0" = 6 IP, "6.1" = 6⅓ IP (parse as int + frac/3)
    .hits / .earnedRuns / .baseOnBalls / .strikeOuts / .homeRuns / .battersFaced
teams.home.teamStats.batting
    .runs / .hits / .baseOnBalls / .strikeOuts
    .atBats / .plateAppearances
    .obp / .slg / .ops           (floats, 0.0 if missing)
teams.home.battingOrder          → [player_id, ...]  batting order (9 starters)
```

IP parsing helper (used across multiple tasks):

```python
def _parse_ip(ip_str) -> float:
    """Convert '6.1' → 6.333, '6.2' → 6.667, '6.0' → 6.0"""
    try:
        parts = str(ip_str).split(".")
        return int(parts[0]) + int(parts[1]) / 3 if len(parts) == 2 else float(ip_str)
    except (ValueError, IndexError):
        return 0.0
```

---

## Task 1: Season pitcher and team batting log fetchers

**Files:**
- Modify: `data.py`
- Modify: `tests/test_data.py`

Add `_parse_ip`, `get_pitcher_logs_season`, and `get_team_batting_logs_season` to `data.py`. These fetch all completed games in a season and build per-start/relief and per-game batting parquet caches.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_data.py`:

```python
from unittest.mock import patch, MagicMock, call
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# _parse_ip
# ---------------------------------------------------------------------------

class TestParseIp:
    def test_full_innings(self):
        from data import _parse_ip
        assert _parse_ip("6.0") == pytest.approx(6.0)

    def test_one_third(self):
        from data import _parse_ip
        assert _parse_ip("6.1") == pytest.approx(6.333, abs=0.01)

    def test_two_thirds(self):
        from data import _parse_ip
        assert _parse_ip("6.2") == pytest.approx(6.667, abs=0.01)

    def test_bad_value_returns_zero(self):
        from data import _parse_ip
        assert _parse_ip(None) == pytest.approx(0.0)
        assert _parse_ip("") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_pitcher_logs_season
# ---------------------------------------------------------------------------

def _make_boxscore(game_id, game_date, home_team_id, away_team_id,
                   home_starter_id, away_starter_id):
    """Minimal boxscore structure for mocking."""
    def pitcher_entry(pid, ip="6.0", h=5, er=2, bb=2, k=6, hr=0, bf=24):
        return {
            "person": {"id": pid, "fullName": f"Pitcher{pid}"},
            "stats": {"pitching": {
                "inningsPitched": ip, "hits": h, "earnedRuns": er,
                "baseOnBalls": bb, "strikeOuts": k, "homeRuns": hr,
                "battersFaced": bf,
            }},
        }
    return {
        "teams": {
            "home": {
                "team": {"id": home_team_id},
                "pitchers": [home_starter_id, home_starter_id + 1],
                "players": {
                    f"ID{home_starter_id}": pitcher_entry(home_starter_id),
                    f"ID{home_starter_id + 1}": pitcher_entry(home_starter_id + 1, ip="2.0", bf=7),
                },
            },
            "away": {
                "team": {"id": away_team_id},
                "pitchers": [away_starter_id],
                "players": {
                    f"ID{away_starter_id}": pitcher_entry(away_starter_id, ip="7.0", k=9, bf=27),
                },
            },
        }
    }


def _make_schedule(game_id, game_date, home_team_id, away_team_id):
    return {
        "dates": [{
            "date": game_date,
            "games": [{
                "gamePk": game_id,
                "status": {"codedGameState": "F"},
                "teams": {
                    "home": {"team": {"id": home_team_id}},
                    "away": {"team": {"id": away_team_id}},
                },
            }]
        }]
    }


class TestGetPitcherLogsSeason:
    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        from data import get_pitcher_logs_season, CACHE_DIR as _CACHE_DIR
        import data as data_mod

        schedule = _make_schedule(700001, "2024-04-01", 147, 139)
        boxscore = _make_boxscore(700001, "2024-04-01", 147, 139, 111, 222)

        def fake_api(endpoint, params=None):
            if "schedule" in endpoint:
                return schedule
            return boxscore

        with patch.object(data_mod, "_mlb_api_get", side_effect=fake_api), \
             patch.object(data_mod, "CACHE_DIR", tmp_path):
            df = get_pitcher_logs_season(2024)

        expected_cols = {"pitcher_id", "pitcher_name", "team_id", "game_date",
                         "game_id", "IP", "H", "ER", "BB", "K", "HR", "BF", "is_start"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 3  # 2 home pitchers + 1 away pitcher

    def test_first_pitcher_is_start(self, tmp_path):
        from data import get_pitcher_logs_season
        import data as data_mod

        schedule = _make_schedule(700001, "2024-04-01", 147, 139)
        boxscore = _make_boxscore(700001, "2024-04-01", 147, 139, 111, 222)

        def fake_api(endpoint, params=None):
            if "schedule" in endpoint:
                return schedule
            return boxscore

        with patch.object(data_mod, "_mlb_api_get", side_effect=fake_api), \
             patch.object(data_mod, "CACHE_DIR", tmp_path):
            df = get_pitcher_logs_season(2024)

        starters = df[df["is_start"] == True]
        assert set(starters["pitcher_id"].tolist()) == {111, 222}

    def test_caches_completed_season(self, tmp_path):
        from data import get_pitcher_logs_season
        import data as data_mod

        schedule = _make_schedule(700001, "2023-04-01", 147, 139)
        boxscore = _make_boxscore(700001, "2023-04-01", 147, 139, 111, 222)

        call_count = [0]
        def fake_api(endpoint, params=None):
            call_count[0] += 1
            if "schedule" in endpoint:
                return schedule
            return boxscore

        with patch.object(data_mod, "_mlb_api_get", side_effect=fake_api), \
             patch.object(data_mod, "CACHE_DIR", tmp_path):
            get_pitcher_logs_season(2023)
            first_count = call_count[0]
            get_pitcher_logs_season(2023)  # second call should use cache
            second_count = call_count[0]

        assert second_count == first_count  # no additional API calls


class TestGetTeamBattingLogsSeason:
    def _make_schedule_with_batting(self, game_id, game_date, home_id, away_id):
        return {
            "dates": [{
                "date": game_date,
                "games": [{
                    "gamePk": game_id,
                    "status": {"codedGameState": "F"},
                    "teams": {
                        "home": {"team": {"id": home_id}},
                        "away": {"team": {"id": away_id}},
                    },
                }]
            }]
        }

    def _make_boxscore_batting(self, home_id, away_id):
        def team_entry(team_id):
            return {
                "team": {"id": team_id},
                "pitchers": [],
                "players": {},
                "teamStats": {
                    "batting": {
                        "runs": 4, "hits": 9, "baseOnBalls": 3,
                        "strikeOuts": 8, "atBats": 33,
                        "plateAppearances": 36,
                        "obp": 0.320, "slg": 0.420, "ops": 0.740,
                    }
                },
            }
        return {"teams": {"home": team_entry(home_id), "away": team_entry(away_id)}}

    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        from data import get_team_batting_logs_season
        import data as data_mod

        schedule = self._make_schedule_with_batting(700001, "2024-04-01", 147, 139)
        boxscore = self._make_boxscore_batting(147, 139)

        def fake_api(endpoint, params=None):
            if "schedule" in endpoint:
                return schedule
            return boxscore

        with patch.object(data_mod, "_mlb_api_get", side_effect=fake_api), \
             patch.object(data_mod, "CACHE_DIR", tmp_path):
            df = get_team_batting_logs_season(2024)

        expected_cols = {"team_id", "game_date", "game_id", "R", "H", "BB", "K",
                         "PA", "OBP", "SLG", "OPS"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 2  # home + away team
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_data.py::TestParseIp tests/test_data.py::TestGetPitcherLogsSeason tests/test_data.py::TestGetTeamBattingLogsSeason -v
```

Expected: `ImportError` or `AttributeError` — `_parse_ip`, `get_pitcher_logs_season`, `get_team_batting_logs_season` not yet defined.

- [ ] **Step 3: Add imports and helper to data.py**

At the top of `data.py`, these imports are already present: `logging`, `time`, `functools`, `date/datetime/timedelta`, `Optional`, `pandas`, `requests`. Also add `CACHE_DIR` to the config import:

```python
from config import (
    MLB_STATS_API_BASE,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    PITCHER_ROLLING_DAYS,
    BULLPEN_ROLLING_DAYS,
    CACHE_DIR,
)
```

Add `_parse_ip` right before the existing `_safe_float` at the bottom of `data.py`:

```python
def _parse_ip(ip_str) -> float:
    """Convert MLB API IP string to decimal innings. '6.1' → 6.333, '6.2' → 6.667."""
    try:
        parts = str(ip_str).split(".")
        return int(parts[0]) + int(parts[1]) / 3 if len(parts) == 2 else float(ip_str)
    except (ValueError, IndexError):
        return 0.0
```

- [ ] **Step 4: Add `get_pitcher_logs_season` to data.py**

Add after the `get_yesterdays_results` function:

```python
def get_pitcher_logs_season(season: int) -> pd.DataFrame:
    """Fetch per-start and relief appearance stats for all pitchers in a season.

    Returns DataFrame with columns:
    pitcher_id, pitcher_name, team_id, game_date, game_id,
    IP, H, ER, BB, K, HR, BF, is_start

    Completed seasons (year < current year) are cached permanently.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"pitcher_logs_{season}.parquet"

    if cache_path.exists() and season < date.today().year:
        logger.info("Loading cached pitcher logs for %d.", season)
        return pd.read_parquet(cache_path)

    logger.info("Fetching pitcher logs for %d season...", season)
    schedule_data = _mlb_api_get(
        "schedule",
        params={"sportId": 1, "season": season, "gameType": "R",
                "hydrate": "team"},
    )

    game_ids: list[tuple[int, str]] = []
    for gd in schedule_data.get("dates", []):
        for g in gd.get("games", []):
            if g.get("status", {}).get("codedGameState") == "F":
                game_ids.append((g["gamePk"], gd["date"]))

    rows = []
    for game_id, game_date in game_ids:
        try:
            boxscore = _mlb_api_get(f"game/{game_id}/boxscore")
        except Exception as e:
            logger.warning("Boxscore fetch failed for game %d: %s", game_id, e)
            continue

        for side in ("home", "away"):
            team_data = boxscore["teams"][side]
            team_id = team_data["team"]["id"]
            pitcher_ids: list[int] = team_data.get("pitchers", [])
            players = team_data.get("players", {})

            for i, pid in enumerate(pitcher_ids):
                player = players.get(f"ID{pid}", {})
                ps = player.get("stats", {}).get("pitching", {})
                if not ps:
                    continue
                rows.append({
                    "pitcher_id": pid,
                    "pitcher_name": player.get("person", {}).get("fullName", ""),
                    "team_id": team_id,
                    "game_date": game_date,
                    "game_id": game_id,
                    "IP": _parse_ip(ps.get("inningsPitched", "0.0")),
                    "H": int(ps.get("hits", 0)),
                    "ER": int(ps.get("earnedRuns", 0)),
                    "BB": int(ps.get("baseOnBalls", 0)),
                    "K": int(ps.get("strikeOuts", 0)),
                    "HR": int(ps.get("homeRuns", 0)),
                    "BF": int(ps.get("battersFaced", 0)),
                    "is_start": (i == 0),
                })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "pitcher_id", "pitcher_name", "team_id", "game_date", "game_id",
        "IP", "H", "ER", "BB", "K", "HR", "BF", "is_start",
    ])

    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    if season < date.today().year:
        df.to_parquet(cache_path, index=False)
        logger.info("Cached pitcher logs for %d (%d rows).", season, len(df))

    return df
```

- [ ] **Step 5: Add `get_team_batting_logs_season` to data.py**

Add immediately after `get_pitcher_logs_season`:

```python
def get_team_batting_logs_season(season: int) -> pd.DataFrame:
    """Fetch per-game team batting stats for all teams in a season.

    Returns DataFrame with columns:
    team_id, game_date, game_id, R, H, BB, K, PA, OBP, SLG, OPS

    Completed seasons cached permanently.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"team_batting_logs_{season}.parquet"

    if cache_path.exists() and season < date.today().year:
        logger.info("Loading cached team batting logs for %d.", season)
        return pd.read_parquet(cache_path)

    logger.info("Fetching team batting logs for %d season...", season)
    schedule_data = _mlb_api_get(
        "schedule",
        params={"sportId": 1, "season": season, "gameType": "R",
                "hydrate": "team"},
    )

    game_ids: list[tuple[int, str]] = []
    for gd in schedule_data.get("dates", []):
        for g in gd.get("games", []):
            if g.get("status", {}).get("codedGameState") == "F":
                game_ids.append((g["gamePk"], gd["date"]))

    rows = []
    for game_id, game_date in game_ids:
        try:
            boxscore = _mlb_api_get(f"game/{game_id}/boxscore")
        except Exception as e:
            logger.warning("Boxscore fetch failed for game %d: %s", game_id, e)
            continue

        for side in ("home", "away"):
            team_data = boxscore["teams"][side]
            team_id = team_data["team"]["id"]
            batting = team_data.get("teamStats", {}).get("batting", {})
            rows.append({
                "team_id": team_id,
                "game_date": game_date,
                "game_id": game_id,
                "R": int(batting.get("runs", 0)),
                "H": int(batting.get("hits", 0)),
                "BB": int(batting.get("baseOnBalls", 0)),
                "K": int(batting.get("strikeOuts", 0)),
                "PA": int(batting.get("plateAppearances", 0)),
                "OBP": float(batting.get("obp", 0.0)),
                "SLG": float(batting.get("slg", 0.0)),
                "OPS": float(batting.get("ops", 0.0)),
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "team_id", "game_date", "game_id", "R", "H", "BB", "K", "PA", "OBP", "SLG", "OPS",
    ])

    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    if season < date.today().year:
        df.to_parquet(cache_path, index=False)
        logger.info("Cached team batting logs for %d (%d rows).", season, len(df))

    return df
```

- [ ] **Step 6: Run tests to confirm they pass**

```
pytest tests/test_data.py::TestParseIp tests/test_data.py::TestGetPitcherLogsSeason tests/test_data.py::TestGetTeamBattingLogsSeason -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add data.py tests/test_data.py
git commit -m "feat: add get_pitcher_logs_season and get_team_batting_logs_season"
```

---

## Task 2: Lineup and live fetch functions

**Files:**
- Modify: `data.py`
- Modify: `tests/test_data.py`

Add `get_lineup_ops_season`, `get_pitcher_recent_logs`, `get_team_recent_batting`.

`get_lineup_ops_season` fetches the boxscore batting order for each historical game and looks up each player's season OPS via `people/{pid}/stats?stats=season&group=hitting&season={year}`, caching per-player-season. Falls back to 0.720 if unavailable.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_data.py`:

```python
class TestGetLineupOpsSeason:
    def _make_schedule(self, game_id, game_date):
        return {
            "dates": [{
                "date": game_date,
                "games": [{"gamePk": game_id, "status": {"codedGameState": "F"},
                            "teams": {"home": {"team": {"id": 147}},
                                      "away": {"team": {"id": 139}}}}]
            }]
        }

    def _make_boxscore_with_order(self, home_ids, away_ids):
        return {
            "teams": {
                "home": {"team": {"id": 147}, "battingOrder": home_ids,
                         "pitchers": [], "players": {}, "teamStats": {"batting": {}}},
                "away": {"team": {"id": 139}, "battingOrder": away_ids,
                         "pitchers": [], "players": {}, "teamStats": {"batting": {}}},
            }
        }

    def _make_player_stats(self, ops=0.800):
        return {"stats": [{"splits": [{"stat": {"ops": ops}}]}]}

    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        from data import get_lineup_ops_season
        import data as data_mod
        import pandas as pd

        home_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        away_ids = [11, 12, 13, 14, 15, 16, 17, 18, 19]
        historical = pd.DataFrame([{
            "game_id": 700001, "game_date": "2024-04-01",
            "home_team_id": 147, "away_team_id": 139,
        }])

        def fake_api(endpoint, params=None):
            if "schedule" in endpoint:
                return self._make_schedule(700001, "2024-04-01")
            if "boxscore" in endpoint:
                return self._make_boxscore_with_order(home_ids, away_ids)
            return self._make_player_stats(0.780)

        with patch.object(data_mod, "_mlb_api_get", side_effect=fake_api), \
             patch.object(data_mod, "CACHE_DIR", tmp_path):
            df = get_lineup_ops_season(2024, historical)

        assert set(df.columns) >= {"game_id", "team_id", "game_date", "lineup_ops"}
        assert len(df) == 2  # home + away team per game

    def test_falls_back_to_default_ops_on_api_error(self, tmp_path):
        from data import get_lineup_ops_season
        import data as data_mod
        import pandas as pd

        historical = pd.DataFrame([{
            "game_id": 700001, "game_date": "2024-04-01",
            "home_team_id": 147, "away_team_id": 139,
        }])

        def fake_api(endpoint, params=None):
            if "schedule" in endpoint:
                return self._make_schedule(700001, "2024-04-01")
            if "boxscore" in endpoint:
                return self._make_boxscore_with_order([1,2,3,4,5,6,7,8,9], [11,12,13,14,15,16,17,18,19])
            raise Exception("API error")

        with patch.object(data_mod, "_mlb_api_get", side_effect=fake_api), \
             patch.object(data_mod, "CACHE_DIR", tmp_path):
            df = get_lineup_ops_season(2024, historical)

        assert (df["lineup_ops"] == 0.720).all()


class TestGetPitcherRecentLogs:
    def test_returns_dataframe(self):
        from data import get_pitcher_recent_logs
        import data as data_mod

        def fake_api(endpoint, params=None):
            return {
                "stats": [{
                    "splits": [
                        {"date": "2024-04-10", "team": {"id": 147},
                         "game": {"gamePk": 700001},
                         "stat": {"inningsPitched": "6.0", "hits": 5, "earnedRuns": 2,
                                  "baseOnBalls": 2, "strikeOuts": 7, "homeRuns": 0,
                                  "battersFaced": 24}},
                    ]
                }]
            }

        with patch.object(data_mod, "_mlb_api_get", side_effect=fake_api):
            df = get_pitcher_recent_logs(123456, n=10)

        assert set(df.columns) >= {"game_date", "IP", "H", "ER", "BB", "K", "HR", "BF", "is_start"}
        assert len(df) == 1

    def test_returns_empty_dataframe_on_error(self):
        from data import get_pitcher_recent_logs
        import data as data_mod

        with patch.object(data_mod, "_mlb_api_get", side_effect=Exception("API down")):
            df = get_pitcher_recent_logs(999, n=10)

        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestGetTeamRecentBatting:
    def test_returns_dataframe(self):
        from data import get_team_recent_batting
        import data as data_mod

        def fake_api(endpoint, params=None):
            return {
                "stats": [{
                    "splits": [
                        {"date": "2024-04-10", "game": {"gamePk": 700001},
                         "stat": {"runs": 5, "hits": 10, "baseOnBalls": 3,
                                  "strikeOuts": 8, "plateAppearances": 36,
                                  "obp": 0.330, "slg": 0.450, "ops": 0.780}},
                    ]
                }]
            }

        with patch.object(data_mod, "_mlb_api_get", side_effect=fake_api):
            df = get_team_recent_batting(147, n=10)

        assert set(df.columns) >= {"game_date", "R", "H", "BB", "K", "PA", "OBP", "SLG", "OPS"}
        assert len(df) == 1

    def test_returns_empty_dataframe_on_error(self):
        from data import get_team_recent_batting
        import data as data_mod

        with patch.object(data_mod, "_mlb_api_get", side_effect=Exception("API down")):
            df = get_team_recent_batting(147, n=10)

        assert isinstance(df, pd.DataFrame)
        assert df.empty
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_data.py::TestGetLineupOpsSeason tests/test_data.py::TestGetPitcherRecentLogs tests/test_data.py::TestGetTeamRecentBatting -v
```

Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Add `get_lineup_ops_season` to data.py**

Add after `get_team_batting_logs_season`:

```python
def get_lineup_ops_season(season: int, historical_games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game lineup OPS for training data.

    For each game in historical_games, fetches the batting order from the
    boxscore and averages each batter's season OPS. Falls back to 0.720
    per batter when stats are unavailable.

    Returns DataFrame with columns: game_id, team_id, game_date, lineup_ops
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"lineups_{season}.parquet"

    if cache_path.exists() and season < date.today().year:
        logger.info("Loading cached lineup OPS for %d.", season)
        return pd.read_parquet(cache_path)

    logger.info("Building lineup OPS for %d (%d games)...", season, len(historical_games))

    # Per-player season OPS cache (avoids re-fetching same player multiple times)
    player_ops_cache: dict[int, float] = {}

    def _get_player_ops(player_id: int) -> float:
        if player_id in player_ops_cache:
            return player_ops_cache[player_id]
        try:
            resp = _mlb_api_get(
                f"people/{player_id}/stats",
                params={"stats": "season", "group": "hitting", "season": season},
            )
            splits = resp.get("stats", [{}])[0].get("splits", [])
            ops = float(splits[0]["stat"]["ops"]) if splits else 0.720
        except Exception:
            ops = 0.720
        player_ops_cache[player_id] = ops
        return ops

    rows = []
    for _, game in historical_games.iterrows():
        game_id = int(game["game_id"])
        game_date = str(game["game_date"])
        home_team_id = int(game["home_team_id"])
        away_team_id = int(game["away_team_id"])

        try:
            boxscore = _mlb_api_get(f"game/{game_id}/boxscore")
        except Exception as e:
            logger.debug("Boxscore failed for game %d: %s. Using default lineup OPS.", game_id, e)
            rows.extend([
                {"game_id": game_id, "team_id": home_team_id, "game_date": game_date, "lineup_ops": 0.720},
                {"game_id": game_id, "team_id": away_team_id, "game_date": game_date, "lineup_ops": 0.720},
            ])
            continue

        for side, team_id in (("home", home_team_id), ("away", away_team_id)):
            batting_order: list[int] = boxscore["teams"][side].get("battingOrder", [])
            if not batting_order:
                ops_val = 0.720
            else:
                ops_vals = [_get_player_ops(pid) for pid in batting_order[:9]]
                ops_val = round(sum(ops_vals) / len(ops_vals), 3)
            rows.append({
                "game_id": game_id,
                "team_id": team_id,
                "game_date": game_date,
                "lineup_ops": ops_val,
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["game_id", "team_id", "game_date", "lineup_ops"]
    )
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    if season < date.today().year:
        df.to_parquet(cache_path, index=False)
        logger.info("Cached lineup OPS for %d (%d rows).", season, len(df))

    return df
```

- [ ] **Step 4: Add `get_pitcher_recent_logs` to data.py**

Add after `get_lineup_ops_season`:

```python
@retry_on_failure
def get_pitcher_recent_logs(pitcher_id: int, n: int = 10) -> pd.DataFrame:
    """Fetch a pitcher's last n appearances from MLB Stats API (prediction path).

    Returns DataFrame with same schema as pitcher_logs parquet:
    game_date, game_id, team_id, IP, H, ER, BB, K, HR, BF, is_start
    """
    try:
        data = _mlb_api_get(
            f"people/{pitcher_id}/stats",
            params={"stats": "gameLog", "group": "pitching",
                    "season": date.today().year},
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        splits = splits[-n:]  # last n appearances

        rows = []
        for split in splits:
            stat = split.get("stat", {})
            rows.append({
                "game_date": pd.to_datetime(split.get("date")).date(),
                "game_id": split.get("game", {}).get("gamePk"),
                "team_id": split.get("team", {}).get("id"),
                "IP": _parse_ip(stat.get("inningsPitched", "0.0")),
                "H": int(stat.get("hits", 0)),
                "ER": int(stat.get("earnedRuns", 0)),
                "BB": int(stat.get("baseOnBalls", 0)),
                "K": int(stat.get("strikeOuts", 0)),
                "HR": int(stat.get("homeRuns", 0)),
                "BF": int(stat.get("battersFaced", 0)),
                "is_start": bool(stat.get("gamesStarted", 0)),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "game_date", "game_id", "team_id", "IP", "H", "ER", "BB", "K", "HR", "BF", "is_start"
        ])
    except Exception as e:
        logger.warning("get_pitcher_recent_logs(%d) failed: %s", pitcher_id, e)
        return pd.DataFrame(columns=[
            "game_date", "game_id", "team_id", "IP", "H", "ER", "BB", "K", "HR", "BF", "is_start"
        ])
```

- [ ] **Step 5: Add `get_team_recent_batting` to data.py**

Add after `get_pitcher_recent_logs`:

```python
@retry_on_failure
def get_team_recent_batting(team_id: int, n: int = 10) -> pd.DataFrame:
    """Fetch a team's last n games' batting stats (prediction path).

    Returns DataFrame with same schema as team_batting_logs parquet:
    game_date, game_id, R, H, BB, K, PA, OBP, SLG, OPS
    """
    try:
        data = _mlb_api_get(
            f"teams/{team_id}/stats",
            params={"stats": "gameLog", "group": "hitting",
                    "season": date.today().year},
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        splits = splits[-n:]

        rows = []
        for split in splits:
            stat = split.get("stat", {})
            rows.append({
                "game_date": pd.to_datetime(split.get("date")).date(),
                "game_id": split.get("game", {}).get("gamePk"),
                "R": int(stat.get("runs", 0)),
                "H": int(stat.get("hits", 0)),
                "BB": int(stat.get("baseOnBalls", 0)),
                "K": int(stat.get("strikeOuts", 0)),
                "PA": int(stat.get("plateAppearances", 0)),
                "OBP": float(stat.get("obp", 0.0)),
                "SLG": float(stat.get("slg", 0.0)),
                "OPS": float(stat.get("ops", 0.0)),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "game_date", "game_id", "R", "H", "BB", "K", "PA", "OBP", "SLG", "OPS"
        ])
    except Exception as e:
        logger.warning("get_team_recent_batting(%d) failed: %s", team_id, e)
        return pd.DataFrame(columns=[
            "game_date", "game_id", "R", "H", "BB", "K", "PA", "OBP", "SLG", "OPS"
        ])
```

- [ ] **Step 6: Run tests to confirm they pass**

```
pytest tests/test_data.py::TestGetLineupOpsSeason tests/test_data.py::TestGetPitcherRecentLogs tests/test_data.py::TestGetTeamRecentBatting -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add data.py tests/test_data.py
git commit -m "feat: add get_lineup_ops_season, get_pitcher_recent_logs, get_team_recent_batting"
```

---

## Task 3: Feature computation functions

**Files:**
- Modify: `features.py` (new FEATURE_COLUMNS, PARK_FACTORS, 4 compute functions)
- Create: `tests/test_features.py`

Replace `FEATURE_COLUMNS` with the 35-column schema and add the four pure computation functions. The existing `build_game_features` and `build_training_features` are left intact for now (replaced in Tasks 4 and 5).

- [ ] **Step 1: Create `tests/test_features.py` with failing tests**

```python
"""Unit tests for feature engineering computation functions."""

import pytest
import pandas as pd
from datetime import date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PITCHER_LOG_COLS = ["pitcher_id", "team_id", "game_date", "IP", "H", "ER", "BB", "K", "BF", "is_start"]
_BATTING_LOG_COLS = ["team_id", "game_date", "OBP", "OPS"]


def make_pitcher_logs(rows):
    """rows = list of (pitcher_id, team_id, game_date_str, IP, H, ER, BB, K, BF, is_start)"""
    if not rows:
        return pd.DataFrame(columns=_PITCHER_LOG_COLS)
    return pd.DataFrame([
        {"pitcher_id": r[0], "team_id": r[1],
         "game_date": date.fromisoformat(r[2]),
         "IP": r[3], "H": r[4], "ER": r[5], "BB": r[6],
         "K": r[7], "BF": r[8], "is_start": r[9]}
        for r in rows
    ])


def make_batting_logs(rows):
    """rows = list of (team_id, game_date_str, OBP, OPS)"""
    if not rows:
        return pd.DataFrame(columns=_BATTING_LOG_COLS)
    return pd.DataFrame([
        {"team_id": r[0], "game_date": date.fromisoformat(r[1]),
         "OBP": r[2], "OPS": r[3]}
        for r in rows
    ])


# ---------------------------------------------------------------------------
# FEATURE_COLUMNS
# ---------------------------------------------------------------------------

class TestFeatureColumns:
    def test_has_35_columns(self):
        from features import FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) == 35

    def test_contains_park_factor(self):
        from features import FEATURE_COLUMNS
        assert "park_factor" in FEATURE_COLUMNS

    def test_contains_rest_and_b2b(self):
        from features import FEATURE_COLUMNS
        assert "home_pitcher_rest_days" in FEATURE_COLUMNS
        assert "away_pitcher_rest_days" in FEATURE_COLUMNS
        assert "home_team_back_to_back" in FEATURE_COLUMNS
        assert "away_team_back_to_back" in FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# compute_pitcher_rolling
# ---------------------------------------------------------------------------

class TestComputePitcherRolling:
    def _make_starts(self, pitcher_id, dates_and_lines):
        """dates_and_lines: list of (date_str, IP, H, ER, BB, K, BF)"""
        return make_pitcher_logs([
            (pitcher_id, 147, d, ip, h, er, bb, k, bf, True)
            for d, ip, h, er, bb, k, bf in dates_and_lines
        ])

    def test_returns_10_values(self):
        from features import compute_pitcher_rolling
        logs = self._make_starts(111, [
            ("2024-04-01", 6.0, 4, 2, 2, 7, 24),
            ("2024-04-07", 7.0, 5, 1, 1, 9, 27),
        ])
        result = compute_pitcher_rolling(logs, 111, "2024-04-14")
        assert len(result) == 10

    def test_correct_key_names(self):
        from features import compute_pitcher_rolling
        logs = self._make_starts(111, [("2024-04-01", 6.0, 4, 2, 2, 7, 24)])
        result = compute_pitcher_rolling(logs, 111, "2024-04-14")
        expected_keys = {
            "ERA_3", "WHIP_3", "K_pct_3", "BB_pct_3", "K_BB_pct_3",
            "ERA_10", "WHIP_10", "K_pct_10", "BB_pct_10", "K_BB_pct_10",
        }
        assert set(result.keys()) == expected_keys

    def test_only_uses_starts_before_date(self):
        from features import compute_pitcher_rolling
        logs = self._make_starts(111, [
            ("2024-04-01", 6.0, 4, 2, 2, 7, 24),
            ("2024-04-10", 6.0, 3, 3, 3, 5, 24),  # this is on or after as_of
        ])
        # as_of_date = April 10, so only April 1 game should be used
        result_before = compute_pitcher_rolling(logs, 111, "2024-04-10")
        result_after = compute_pitcher_rolling(logs, 111, "2024-04-11")
        # ERA should differ because April 10 game is excluded in before case
        assert result_before["ERA_3"] != result_after["ERA_3"]

    def test_era_calculation(self):
        from features import compute_pitcher_rolling
        # 2 ER in 6 IP → ERA = 2 * 9 / 6 = 3.00
        logs = self._make_starts(111, [("2024-04-01", 6.0, 4, 2, 2, 7, 24)])
        result = compute_pitcher_rolling(logs, 111, "2024-04-08")
        assert result["ERA_3"] == pytest.approx(3.0, abs=0.01)

    def test_k_pct_calculation(self):
        from features import compute_pitcher_rolling
        # 7 K in 24 BF → K% = 7/24 = 0.292
        logs = self._make_starts(111, [("2024-04-01", 6.0, 4, 2, 2, 7, 24)])
        result = compute_pitcher_rolling(logs, 111, "2024-04-08")
        assert result["K_pct_3"] == pytest.approx(7 / 24, abs=0.001)

    def test_default_when_no_starts(self):
        from features import compute_pitcher_rolling
        logs = make_pitcher_logs([])
        result = compute_pitcher_rolling(logs, 999, "2024-04-08")
        assert result["ERA_3"] == pytest.approx(4.20)
        assert result["WHIP_3"] == pytest.approx(1.30)
        assert result["K_pct_3"] == pytest.approx(0.21)
        assert result["BB_pct_3"] == pytest.approx(0.08)

    def test_uses_all_available_when_fewer_than_window(self):
        from features import compute_pitcher_rolling
        # Only 2 starts available, window=3; should use both
        logs = self._make_starts(111, [
            ("2024-04-01", 6.0, 4, 2, 2, 7, 24),
            ("2024-04-07", 6.0, 4, 3, 3, 5, 24),
        ])
        result = compute_pitcher_rolling(logs, 111, "2024-04-14")
        # ERA = (2+3)*9 / (6+6) = 45/12 = 3.75
        assert result["ERA_3"] == pytest.approx(3.75, abs=0.01)

    def test_10_window_uses_last_10_not_all(self):
        from features import compute_pitcher_rolling
        # 12 starts, window=10 should use last 10 only
        rows = [
            (f"2024-0{i+1}-01" if i < 9 else f"2024-{i+1:02d}-01",
             6.0, 4, 1, 2, 7, 24)  # 1 ER each
            for i in range(10)
        ]
        rows.insert(0, ("2024-01-01", 6.0, 4, 9, 2, 7, 24))  # outlier ER=9, oldest
        rows.insert(1, ("2024-01-15", 6.0, 4, 9, 2, 7, 24))  # outlier ER=9, second oldest
        logs = self._make_starts(111, rows)
        result = compute_pitcher_rolling(logs, 111, "2024-12-01")
        # If using only last 10: ERA = 1*9/6 = 1.50; if using all 12: much higher
        assert result["ERA_10"] == pytest.approx(1.50, abs=0.01)


# ---------------------------------------------------------------------------
# compute_bullpen_rolling
# ---------------------------------------------------------------------------

class TestComputeBullpenRolling:
    def test_returns_2_values(self):
        from features import compute_bullpen_rolling
        logs = make_pitcher_logs([
            (500, 147, "2024-04-01", 2.0, 2, 1, 1, 2, 8, False),
        ])
        result = compute_bullpen_rolling(logs, 147, "2024-04-08")
        assert len(result) == 2

    def test_correct_key_names(self):
        from features import compute_bullpen_rolling
        logs = make_pitcher_logs([])
        result = compute_bullpen_rolling(logs, 147, "2024-04-08")
        assert set(result.keys()) == {"ERA_10", "WHIP_10"}

    def test_only_uses_relief_appearances(self):
        from features import compute_bullpen_rolling
        # Mix of starter and reliever appearances for team 147
        logs = make_pitcher_logs([
            (100, 147, "2024-04-01", 6.0, 4, 0, 1, 7, 22, True),   # starter (is_start=True)
            (200, 147, "2024-04-01", 2.0, 1, 1, 0, 2, 7, False),    # reliever
        ])
        # With only reliever: 1 ER in 2 IP → ERA = 4.50
        result = compute_bullpen_rolling(logs, 147, "2024-04-08")
        assert result["ERA_10"] == pytest.approx(4.50, abs=0.01)

    def test_filters_by_team_id(self):
        from features import compute_bullpen_rolling
        logs = make_pitcher_logs([
            (200, 147, "2024-04-01", 2.0, 1, 1, 0, 2, 7, False),
            (300, 139, "2024-04-01", 2.0, 1, 3, 0, 2, 7, False),  # different team
        ])
        result_147 = compute_bullpen_rolling(logs, 147, "2024-04-08")
        result_139 = compute_bullpen_rolling(logs, 139, "2024-04-08")
        assert result_147["ERA_10"] != result_139["ERA_10"]

    def test_default_when_no_appearances(self):
        from features import compute_bullpen_rolling
        result = compute_bullpen_rolling(make_pitcher_logs([]), 147, "2024-04-08")
        assert result["ERA_10"] == pytest.approx(4.20)
        assert result["WHIP_10"] == pytest.approx(1.30)


# ---------------------------------------------------------------------------
# compute_team_offense_rolling
# ---------------------------------------------------------------------------

class TestComputeTeamOffenseRolling:
    def test_returns_2_values(self):
        from features import compute_team_offense_rolling
        logs = make_batting_logs([(147, "2024-04-01", 0.320, 0.740)])
        result = compute_team_offense_rolling(logs, 147, "2024-04-08")
        assert len(result) == 2

    def test_correct_key_names(self):
        from features import compute_team_offense_rolling
        result = compute_team_offense_rolling(make_batting_logs([]), 147, "2024-04-08")
        assert set(result.keys()) == {"OPS_10", "OBP_10"}

    def test_averages_last_10_games(self):
        from features import compute_team_offense_rolling
        rows = [(147, f"2024-04-{i+1:02d}", 0.320, 0.740) for i in range(10)]
        logs = make_batting_logs(rows)
        result = compute_team_offense_rolling(logs, 147, "2024-04-15")
        assert result["OPS_10"] == pytest.approx(0.740, abs=0.001)
        assert result["OBP_10"] == pytest.approx(0.320, abs=0.001)

    def test_only_uses_games_before_date(self):
        from features import compute_team_offense_rolling
        logs = make_batting_logs([
            (147, "2024-04-01", 0.320, 0.740),
            (147, "2024-04-08", 0.400, 0.900),  # on the as_of date — excluded
        ])
        result = compute_team_offense_rolling(logs, 147, "2024-04-08")
        assert result["OPS_10"] == pytest.approx(0.740, abs=0.001)

    def test_default_when_fewer_than_3_games(self):
        from features import compute_team_offense_rolling
        result = compute_team_offense_rolling(make_batting_logs([]), 147, "2024-04-08")
        assert result["OPS_10"] == pytest.approx(0.720)
        assert result["OBP_10"] == pytest.approx(0.315)

    def test_filters_by_team_id(self):
        from features import compute_team_offense_rolling
        logs = make_batting_logs([
            (147, "2024-04-01", 0.350, 0.800),
            (139, "2024-04-01", 0.290, 0.650),
        ])
        r147 = compute_team_offense_rolling(logs, 147, "2024-04-08")
        r139 = compute_team_offense_rolling(logs, 139, "2024-04-08")
        assert r147["OPS_10"] != r139["OPS_10"]


# ---------------------------------------------------------------------------
# compute_rest
# ---------------------------------------------------------------------------

class TestComputeRest:
    def test_correct_key_names(self):
        from features import compute_rest
        p_logs = make_pitcher_logs([])
        b_logs = make_batting_logs([])
        result = compute_rest(p_logs, 111, b_logs, 147, "2024-04-08")
        assert set(result.keys()) == {"rest_days", "team_back_to_back"}

    def test_rest_days_calculated_correctly(self):
        from features import compute_rest
        # Last start was April 1, game is April 6 → 5 rest days
        p_logs = make_pitcher_logs([
            (111, 147, "2024-04-01", 6.0, 4, 2, 2, 7, 24, True),
        ])
        result = compute_rest(p_logs, 111, make_batting_logs([]), 147, "2024-04-06")
        assert result["rest_days"] == 5

    def test_rest_days_default_when_no_prior_start(self):
        from features import compute_rest
        result = compute_rest(make_pitcher_logs([]), 999, make_batting_logs([]), 147, "2024-04-08")
        assert result["rest_days"] == 4

    def test_back_to_back_true(self):
        from features import compute_rest
        # Team played yesterday (April 7 → April 8 game)
        b_logs = make_batting_logs([(147, "2024-04-07", 0.320, 0.740)])
        result = compute_rest(make_pitcher_logs([]), 111, b_logs, 147, "2024-04-08")
        assert result["team_back_to_back"] == 1

    def test_back_to_back_false(self):
        from features import compute_rest
        # Last game was 2 days ago
        b_logs = make_batting_logs([(147, "2024-04-06", 0.320, 0.740)])
        result = compute_rest(make_pitcher_logs([]), 111, b_logs, 147, "2024-04-08")
        assert result["team_back_to_back"] == 0

    def test_only_uses_starts_for_pitcher_rest(self):
        from features import compute_rest
        # Pitcher has a relief appearance more recently but starter appearance further back
        p_logs = make_pitcher_logs([
            (111, 147, "2024-04-01", 6.0, 4, 2, 2, 7, 24, True),   # start
            (111, 147, "2024-04-05", 1.0, 0, 0, 0, 1, 4, False),   # relief
        ])
        # Rest should be from last START (April 1), not last relief (April 5)
        result = compute_rest(p_logs, 111, make_batting_logs([]), 147, "2024-04-08")
        assert result["rest_days"] == 7  # Apr 8 - Apr 1


# ---------------------------------------------------------------------------
# PARK_FACTORS
# ---------------------------------------------------------------------------

class TestParkFactors:
    def test_coors_is_highest(self):
        from features import PARK_FACTORS
        assert PARK_FACTORS["COL"] > 1.20

    def test_neutral_park_is_1(self):
        from features import PARK_FACTORS
        assert PARK_FACTORS.get("LAD") == pytest.approx(1.0)

    def test_covers_all_30_teams(self):
        from features import PARK_FACTORS
        assert len(PARK_FACTORS) >= 30
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_features.py -v
```

Expected: `ImportError` — `compute_pitcher_rolling` etc. not yet defined.

- [ ] **Step 3: Update imports in features.py**

Replace the existing import block at the top of `features.py` with:

```python
"""Feature engineering — build sabermetric feature vectors for model input."""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
```

(The old imports — `numpy`, `data.get_pitcher_stats`, etc. — are all removed here. `get_pitcher_recent_logs` and `get_team_recent_batting` are imported later in Task 5.)

- [ ] **Step 4: Replace FEATURE_COLUMNS and add PARK_FACTORS in features.py**

Replace the existing `FEATURE_COLUMNS` list (lines 21–35) with:

```python
FEATURE_COLUMNS = [
    # Home starting pitcher — last 3 starts
    "home_p_ERA_3", "home_p_WHIP_3", "home_p_K_pct_3", "home_p_BB_pct_3", "home_p_K_BB_pct_3",
    # Home starting pitcher — last 10 starts
    "home_p_ERA_10", "home_p_WHIP_10", "home_p_K_pct_10", "home_p_BB_pct_10", "home_p_K_BB_pct_10",
    # Away starting pitcher — last 3 starts
    "away_p_ERA_3", "away_p_WHIP_3", "away_p_K_pct_3", "away_p_BB_pct_3", "away_p_K_BB_pct_3",
    # Away starting pitcher — last 10 starts
    "away_p_ERA_10", "away_p_WHIP_10", "away_p_K_pct_10", "away_p_BB_pct_10", "away_p_K_BB_pct_10",
    # Bullpen (last 10 relief appearances)
    "home_bullpen_era_10", "home_bullpen_whip_10",
    "away_bullpen_era_10", "away_bullpen_whip_10",
    # Team offense (rolling 10 games)
    "home_team_ops_10", "home_team_obp_10",
    "away_team_ops_10", "away_team_obp_10",
    # Lineup quality
    "home_lineup_ops", "away_lineup_ops",
    # Park factor
    "park_factor",
    # Rest / fatigue
    "home_pitcher_rest_days", "away_pitcher_rest_days",
    "home_team_back_to_back", "away_team_back_to_back",
]

PARK_FACTORS: dict[str, float] = {
    "COL": 1.24,
    "CIN": 1.10, "TEX": 1.09, "BOS": 1.08, "PHI": 1.07,
    "MIL": 1.05, "CHC": 1.04, "ATL": 1.03, "HOU": 1.02,
    "BAL": 1.02, "NYY": 1.01, "TOR": 1.01, "LAD": 1.00,
    "STL": 1.00, "WSH": 0.99, "MIN": 0.99, "DET": 0.99,
    "CLE": 0.98, "AZ":  0.98, "NYM": 0.98, "CHW": 0.97,
    "KC":  0.97, "LAA": 0.97, "SF":  0.96, "SEA": 0.96,
    "PIT": 0.96, "TB":  0.95, "MIA": 0.95, "SD":  0.94,
    "ATH": 0.93, "OAK": 0.93,
}
```

- [ ] **Step 5: Add `compute_pitcher_rolling` to features.py**

Add after the `PARK_FACTORS` dict:

```python
def compute_pitcher_rolling(
    logs: pd.DataFrame,
    pitcher_id: int,
    as_of_date: str,
) -> dict:
    """Compute rolling ERA/WHIP/K%/BB%/K-BB% for last 3 and last 10 starts.

    Args:
        logs: pitcher_logs DataFrame (all pitchers, all games in season).
        pitcher_id: The pitcher to compute stats for.
        as_of_date: ISO date string; only starts BEFORE this date are used.

    Returns:
        Dict with 10 keys: ERA_3, WHIP_3, K_pct_3, BB_pct_3, K_BB_pct_3,
                           ERA_10, WHIP_10, K_pct_10, BB_pct_10, K_BB_pct_10
    """
    cutoff = date.fromisoformat(as_of_date) if isinstance(as_of_date, str) else as_of_date

    if logs.empty or "pitcher_id" not in logs.columns:
        return {"ERA_3": 4.20, "WHIP_3": 1.30, "K_pct_3": 0.21, "BB_pct_3": 0.08, "K_BB_pct_3": 0.13,
                "ERA_10": 4.20, "WHIP_10": 1.30, "K_pct_10": 0.21, "BB_pct_10": 0.08, "K_BB_pct_10": 0.13}

    mask = (
        (logs["pitcher_id"] == pitcher_id) &
        (logs["is_start"] == True) &
        (logs["game_date"] < cutoff)
    )
    starts = logs[mask].sort_values("game_date")

    def _stats_from_slice(df: pd.DataFrame) -> dict:
        if df.empty:
            logger.debug("No starts found for pitcher %d before %s — using defaults.", pitcher_id, as_of_date)
            return {"ERA": 4.20, "WHIP": 1.30, "K_pct": 0.21, "BB_pct": 0.08, "K_BB_pct": 0.13}
        total_ip = df["IP"].sum()
        if total_ip == 0:
            return {"ERA": 4.20, "WHIP": 1.30, "K_pct": 0.21, "BB_pct": 0.08, "K_BB_pct": 0.13}
        total_bf = df["BF"].sum()
        era = df["ER"].sum() * 9 / total_ip
        whip = (df["H"].sum() + df["BB"].sum()) / total_ip
        k_pct = df["K"].sum() / total_bf if total_bf > 0 else 0.21
        bb_pct = df["BB"].sum() / total_bf if total_bf > 0 else 0.08
        return {
            "ERA": round(era, 3),
            "WHIP": round(whip, 3),
            "K_pct": round(k_pct, 4),
            "BB_pct": round(bb_pct, 4),
            "K_BB_pct": round(k_pct - bb_pct, 4),
        }

    last_3 = _stats_from_slice(starts.tail(3))
    last_10 = _stats_from_slice(starts.tail(10))

    return {
        "ERA_3":    last_3["ERA"],   "WHIP_3":   last_3["WHIP"],
        "K_pct_3":  last_3["K_pct"], "BB_pct_3": last_3["BB_pct"],
        "K_BB_pct_3": last_3["K_BB_pct"],
        "ERA_10":   last_10["ERA"],  "WHIP_10":  last_10["WHIP"],
        "K_pct_10": last_10["K_pct"],"BB_pct_10":last_10["BB_pct"],
        "K_BB_pct_10": last_10["K_BB_pct"],
    }
```

- [ ] **Step 6: Add `compute_bullpen_rolling` to features.py**

Add after `compute_pitcher_rolling`:

```python
def compute_bullpen_rolling(
    logs: pd.DataFrame,
    team_id: int,
    as_of_date: str,
) -> dict:
    """Compute rolling ERA/WHIP from last 10 relief appearances for a team.

    Returns: {"ERA_10": float, "WHIP_10": float}
    """
    cutoff = date.fromisoformat(as_of_date) if isinstance(as_of_date, str) else as_of_date

    if logs.empty or "team_id" not in logs.columns:
        return {"ERA_10": 4.20, "WHIP_10": 1.30}

    mask = (
        (logs["team_id"] == team_id) &
        (logs["is_start"] == False) &
        (logs["game_date"] < cutoff)
    )
    relief = logs[mask].sort_values("game_date").tail(10)

    if relief.empty:
        logger.debug("No relief appearances for team %d before %s — using defaults.", team_id, as_of_date)
        return {"ERA_10": 4.20, "WHIP_10": 1.30}

    total_ip = relief["IP"].sum()
    if total_ip == 0:
        return {"ERA_10": 4.20, "WHIP_10": 1.30}

    era = relief["ER"].sum() * 9 / total_ip
    whip = (relief["H"].sum() + relief["BB"].sum()) / total_ip
    return {"ERA_10": round(era, 3), "WHIP_10": round(whip, 3)}
```

- [ ] **Step 7: Add `compute_team_offense_rolling` to features.py**

Add after `compute_bullpen_rolling`:

```python
def compute_team_offense_rolling(
    logs: pd.DataFrame,
    team_id: int,
    as_of_date: str,
) -> dict:
    """Compute rolling OPS/OBP from last 10 games for a team.

    Returns: {"OPS_10": float, "OBP_10": float}
    """
    cutoff = date.fromisoformat(as_of_date) if isinstance(as_of_date, str) else as_of_date

    if logs.empty or "team_id" not in logs.columns:
        return {"OPS_10": 0.720, "OBP_10": 0.315}

    mask = (logs["team_id"] == team_id) & (logs["game_date"] < cutoff)
    recent = logs[mask].sort_values("game_date").tail(10)

    if len(recent) < 3:
        logger.debug("Fewer than 3 batting games for team %d before %s — using defaults.", team_id, as_of_date)
        return {"OPS_10": 0.720, "OBP_10": 0.315}

    return {
        "OPS_10": round(recent["OPS"].mean(), 3),
        "OBP_10": round(recent["OBP"].mean(), 3),
    }
```

- [ ] **Step 8: Add `compute_rest` to features.py**

Add after `compute_team_offense_rolling`:

```python
def compute_rest(
    pitcher_logs: pd.DataFrame,
    pitcher_id: int,
    team_batting_logs: pd.DataFrame,
    team_id: int,
    game_date: str,
) -> dict:
    """Compute pitcher rest days and team back-to-back flag.

    Returns: {"rest_days": int, "team_back_to_back": int}
    """
    cutoff = date.fromisoformat(game_date) if isinstance(game_date, str) else game_date
    yesterday = cutoff - timedelta(days=1)

    # Pitcher rest: days since last start before game_date
    rest_days = 4  # median default
    if not pitcher_logs.empty and "pitcher_id" in pitcher_logs.columns:
        prior_starts = pitcher_logs[
            (pitcher_logs["pitcher_id"] == pitcher_id) &
            (pitcher_logs["is_start"] == True) &
            (pitcher_logs["game_date"] < cutoff)
        ]
        if not prior_starts.empty:
            last_start = prior_starts["game_date"].max()
            rest_days = (cutoff - last_start).days

    # Back-to-back: did team play yesterday?
    b2b = 0
    if not team_batting_logs.empty and "team_id" in team_batting_logs.columns:
        played_yesterday = team_batting_logs[
            (team_batting_logs["team_id"] == team_id) &
            (team_batting_logs["game_date"] == yesterday)
        ]
        b2b = 1 if not played_yesterday.empty else 0

    return {"rest_days": rest_days, "team_back_to_back": b2b}
```

- [ ] **Step 9: Run all feature tests**

```
pytest tests/test_features.py -v
```

Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add features.py tests/test_features.py
git commit -m "feat: add FEATURE_COLUMNS v2, PARK_FACTORS, and 4 compute functions"
```

---

## Task 4: Rewrite `build_training_features`

**Files:**
- Modify: `features.py`
- Modify: `tests/test_features.py`

Replace the existing `build_training_features` with the new signature that accepts pre-loaded DataFrames. The old function made per-game API calls; the new one is pure computation.

- [ ] **Step 1: Add failing tests to `tests/test_features.py`**

```python
# ---------------------------------------------------------------------------
# build_training_features
# ---------------------------------------------------------------------------

class TestBuildTrainingFeatures:
    def _make_historical(self):
        return pd.DataFrame([{
            "game_id": 700001,
            "game_date": "2024-04-10",
            "season": 2024,
            "home_team": "NYY",
            "away_team": "BOS",
            "home_team_id": 147,
            "away_team_id": 111,
            "home_pitcher_id": 1000,
            "away_pitcher_id": 2000,
            "home_win": 1,
        }])

    def _make_pitcher_logs(self):
        rows = []
        for pid, team_id in ((1000, 147), (2000, 111)):
            for i in range(10):
                rows.append((pid, team_id, f"2024-03-{i+1:02d}",
                             6.0, 4, 2, 2, 7, 24, True))
                rows.append((pid + 100, team_id, f"2024-03-{i+1:02d}",
                             2.0, 1, 1, 1, 2, 8, False))
        return make_pitcher_logs(rows)

    def _make_batting_logs(self):
        rows = []
        for team_id in (147, 111):
            for i in range(10):
                rows.append((team_id, f"2024-03-{i+1:02d}", 0.320, 0.740))
        return make_batting_logs(rows)

    def _make_lineups(self):
        return pd.DataFrame([
            {"game_id": 700001, "team_id": 147, "game_date": "2024-04-10", "lineup_ops": 0.750},
            {"game_id": 700001, "team_id": 111, "game_date": "2024-04-10", "lineup_ops": 0.720},
        ])

    def test_returns_correct_shape(self):
        from features import build_training_features, FEATURE_COLUMNS
        historical = self._make_historical()
        X, y = build_training_features(
            historical,
            self._make_pitcher_logs(),
            self._make_batting_logs(),
            self._make_lineups(),
        )
        assert list(X.columns) == FEATURE_COLUMNS
        assert len(X) == 1
        assert len(y) == 1

    def test_y_is_home_win(self):
        from features import build_training_features
        historical = self._make_historical()
        _, y = build_training_features(
            historical,
            self._make_pitcher_logs(),
            self._make_batting_logs(),
            self._make_lineups(),
        )
        assert y.iloc[0] == 1

    def test_park_factor_uses_home_team(self):
        from features import build_training_features, PARK_FACTORS
        historical = self._make_historical()
        X, _ = build_training_features(
            historical,
            self._make_pitcher_logs(),
            self._make_batting_logs(),
            self._make_lineups(),
        )
        assert X["park_factor"].iloc[0] == pytest.approx(PARK_FACTORS.get("NYY", 1.0))

    def test_lineup_ops_falls_back_to_team_ops_when_missing(self):
        from features import build_training_features
        historical = self._make_historical()
        # Empty lineups → should fall back to team_ops_10
        empty_lineups = pd.DataFrame(columns=["game_id", "team_id", "game_date", "lineup_ops"])
        X, _ = build_training_features(
            historical,
            self._make_pitcher_logs(),
            self._make_batting_logs(),
            empty_lineups,
        )
        # home_lineup_ops should equal home_team_ops_10 (the fallback)
        assert X["home_lineup_ops"].iloc[0] == pytest.approx(X["home_team_ops_10"].iloc[0])

    def test_returns_empty_for_empty_input(self):
        from features import build_training_features, FEATURE_COLUMNS
        X, y = build_training_features(
            pd.DataFrame(),
            make_pitcher_logs([]),
            make_batting_logs([]),
            pd.DataFrame(columns=["game_id", "team_id", "game_date", "lineup_ops"]),
        )
        assert list(X.columns) == FEATURE_COLUMNS
        assert len(X) == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_features.py::TestBuildTrainingFeatures -v
```

Expected: FAIL — `build_training_features` still has old signature.

- [ ] **Step 3: Replace `build_training_features` in features.py**

Replace the entire existing `build_training_features` function with:

```python
def build_training_features(
    historical_games: pd.DataFrame,
    pitcher_logs: pd.DataFrame,
    team_batting_logs: pd.DataFrame,
    lineups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix from historical game data using pre-loaded game logs.

    Processes games in date order. All rolling stats are computed from only
    prior games — no data leakage.

    Args:
        historical_games: DataFrame from get_historical_game_data().
        pitcher_logs: Combined pitcher logs DataFrame (all seasons).
        team_batting_logs: Combined team batting logs DataFrame (all seasons).
        lineups: Per-game lineup OPS DataFrame from get_lineup_ops_season().

    Returns:
        (X, y) — feature DataFrame and target Series (1 = home win).
    """
    if historical_games.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series([], dtype=int, name="home_win")

    logger.info("Building training features for %d games...", len(historical_games))

    # Normalize game_date to date objects for consistent comparison
    games = historical_games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    games = games.sort_values("game_date").reset_index(drop=True)

    if not pitcher_logs.empty:
        pitcher_logs = pitcher_logs.copy()
        pitcher_logs["game_date"] = pd.to_datetime(pitcher_logs["game_date"]).dt.date
    if not team_batting_logs.empty:
        team_batting_logs = team_batting_logs.copy()
        team_batting_logs["game_date"] = pd.to_datetime(team_batting_logs["game_date"]).dt.date

    # Build lineup lookup: (game_id, team_id) → lineup_ops
    lineup_lookup: dict[tuple, float] = {}
    if not lineups.empty:
        for _, row in lineups.iterrows():
            lineup_lookup[(int(row["game_id"]), int(row["team_id"]))] = float(row["lineup_ops"])

    feature_rows = []
    targets = []

    for idx, game in games.iterrows():
        if idx % 500 == 0:
            logger.info("Processing game %d / %d...", idx, len(games))

        game_date_str = str(game["game_date"])
        home_pid = int(game["home_pitcher_id"]) if pd.notna(game.get("home_pitcher_id")) else -1
        away_pid = int(game["away_pitcher_id"]) if pd.notna(game.get("away_pitcher_id")) else -1
        home_tid = int(game["home_team_id"])
        away_tid = int(game["away_team_id"])
        game_id  = int(game["game_id"])

        # Pitcher rolling stats
        hp = compute_pitcher_rolling(pitcher_logs, home_pid, game_date_str)
        ap = compute_pitcher_rolling(pitcher_logs, away_pid, game_date_str)

        # Bullpen rolling stats
        hbp = compute_bullpen_rolling(pitcher_logs, home_tid, game_date_str)
        abp = compute_bullpen_rolling(pitcher_logs, away_tid, game_date_str)

        # Team offense rolling
        hoff = compute_team_offense_rolling(team_batting_logs, home_tid, game_date_str)
        aoff = compute_team_offense_rolling(team_batting_logs, away_tid, game_date_str)

        # Park factor
        park = PARK_FACTORS.get(str(game["home_team"]), 1.0)

        # Rest / fatigue
        hrest = compute_rest(pitcher_logs, home_pid, team_batting_logs, home_tid, game_date_str)
        arest = compute_rest(pitcher_logs, away_pid, team_batting_logs, away_tid, game_date_str)

        # Lineup OPS (fall back to team rolling OPS if unavailable)
        home_lineup = lineup_lookup.get((game_id, home_tid), hoff["OPS_10"])
        away_lineup = lineup_lookup.get((game_id, away_tid), aoff["OPS_10"])

        row = {
            "home_p_ERA_3":    hp["ERA_3"],   "home_p_WHIP_3":   hp["WHIP_3"],
            "home_p_K_pct_3":  hp["K_pct_3"], "home_p_BB_pct_3": hp["BB_pct_3"],
            "home_p_K_BB_pct_3": hp["K_BB_pct_3"],
            "home_p_ERA_10":   hp["ERA_10"],  "home_p_WHIP_10":  hp["WHIP_10"],
            "home_p_K_pct_10": hp["K_pct_10"],"home_p_BB_pct_10":hp["BB_pct_10"],
            "home_p_K_BB_pct_10": hp["K_BB_pct_10"],
            "away_p_ERA_3":    ap["ERA_3"],   "away_p_WHIP_3":   ap["WHIP_3"],
            "away_p_K_pct_3":  ap["K_pct_3"], "away_p_BB_pct_3": ap["BB_pct_3"],
            "away_p_K_BB_pct_3": ap["K_BB_pct_3"],
            "away_p_ERA_10":   ap["ERA_10"],  "away_p_WHIP_10":  ap["WHIP_10"],
            "away_p_K_pct_10": ap["K_pct_10"],"away_p_BB_pct_10":ap["BB_pct_10"],
            "away_p_K_BB_pct_10": ap["K_BB_pct_10"],
            "home_bullpen_era_10":  hbp["ERA_10"], "home_bullpen_whip_10": hbp["WHIP_10"],
            "away_bullpen_era_10":  abp["ERA_10"], "away_bullpen_whip_10": abp["WHIP_10"],
            "home_team_ops_10": hoff["OPS_10"], "home_team_obp_10": hoff["OBP_10"],
            "away_team_ops_10": aoff["OPS_10"], "away_team_obp_10": aoff["OBP_10"],
            "home_lineup_ops": home_lineup, "away_lineup_ops": away_lineup,
            "park_factor": park,
            "home_pitcher_rest_days": hrest["rest_days"],
            "away_pitcher_rest_days": arest["rest_days"],
            "home_team_back_to_back": hrest["team_back_to_back"],
            "away_team_back_to_back": arest["team_back_to_back"],
        }
        feature_rows.append(row)
        targets.append(int(game["home_win"]))

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    y = pd.Series(targets, name="home_win")

    logger.info(
        "Built feature matrix: %d games x %d features. Home win rate: %.1f%%",
        len(X), len(FEATURE_COLUMNS), y.mean() * 100,
    )
    return X, y
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_features.py::TestBuildTrainingFeatures -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add features.py tests/test_features.py
git commit -m "feat: rewrite build_training_features with rolling game-log features"
```

---

## Task 5: Update `build_game_features`

**Files:**
- Modify: `features.py`
- Modify: `tests/test_features.py`

Replace `build_game_features` to use the same compute functions via live fetch (`get_pitcher_recent_logs`, `get_team_recent_batting`). The game dict from `get_todays_games()` already contains `home_team_id`, `away_team_id`, `home_pitcher_id`, `away_pitcher_id`, `home_team`.

- [ ] **Step 1: Add failing tests to `tests/test_features.py`**

```python
# ---------------------------------------------------------------------------
# build_game_features
# ---------------------------------------------------------------------------

class TestBuildGameFeatures:
    def _make_game(self):
        return {
            "game_id": 700001,
            "game_date": "2024-04-10",
            "home_team": "NYY",
            "away_team": "BOS",
            "home_team_id": 147,
            "away_team_id": 111,
            "home_pitcher_id": 1000,
            "home_pitcher_name": "Gerrit Cole",
            "away_pitcher_id": 2000,
            "away_pitcher_name": "Brayan Bello",
            "home_pitcher_hand": "R",
            "away_pitcher_hand": "R",
        }

    def _make_recent_logs(self):
        rows = []
        for i in range(5):
            rows.append({
                "game_date": date(2024, 3, i + 1),
                "game_id": 600000 + i,
                "team_id": 147,
                "IP": 6.0, "H": 4, "ER": 2, "BB": 2, "K": 7, "HR": 0, "BF": 24,
                "is_start": True,
            })
        return pd.DataFrame(rows)

    def _make_recent_batting(self):
        rows = []
        for i in range(5):
            rows.append({
                "game_date": date(2024, 3, i + 1),
                "game_id": 600000 + i,
                "R": 4, "H": 9, "BB": 3, "K": 8,
                "PA": 36, "OBP": 0.320, "SLG": 0.420, "OPS": 0.740,
            })
        return pd.DataFrame(rows)

    def test_returns_dict_with_all_feature_columns(self):
        from features import build_game_features, FEATURE_COLUMNS
        import features as feat_mod

        with patch.object(feat_mod, "get_pitcher_recent_logs", return_value=self._make_recent_logs()), \
             patch.object(feat_mod, "get_team_recent_batting", return_value=self._make_recent_batting()):
            result = build_game_features(self._make_game())

        assert result is not None
        assert set(result.keys()) == set(FEATURE_COLUMNS)

    def test_returns_none_on_exception(self):
        from features import build_game_features
        import features as feat_mod

        with patch.object(feat_mod, "get_pitcher_recent_logs", side_effect=Exception("API down")):
            result = build_game_features(self._make_game())

        assert result is None

    def test_park_factor_correct(self):
        from features import build_game_features, PARK_FACTORS
        import features as feat_mod

        with patch.object(feat_mod, "get_pitcher_recent_logs", return_value=self._make_recent_logs()), \
             patch.object(feat_mod, "get_team_recent_batting", return_value=self._make_recent_batting()):
            result = build_game_features(self._make_game())

        assert result["park_factor"] == pytest.approx(PARK_FACTORS.get("NYY", 1.0))
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_features.py::TestBuildGameFeatures -v
```

Expected: FAIL — `build_game_features` still uses old API calls and `get_pitcher_recent_logs` not imported.

- [ ] **Step 3: Add live-fetch imports to features.py**

Task 3 already updated the import block. Now add the live-fetch imports to it. The import block at the top of `features.py` should look like:

```python
"""Feature engineering — build sabermetric feature vectors for model input."""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from data import get_pitcher_recent_logs, get_team_recent_batting

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Replace `build_game_features` in features.py**

Replace the entire existing `build_game_features` function:

```python
def build_game_features(game: dict) -> Optional[dict]:
    """Build a feature vector for a single game matchup (prediction path).

    Fetches live rolling stats from MLB Stats API using the same computation
    functions as the training path — no train/serve skew.

    Args:
        game: Dict from get_todays_games() — must include home/away team,
              pitcher IDs, team IDs.

    Returns:
        Dict of feature name → value, or None if a critical fetch fails.
    """
    try:
        today = str(date.today())
        home_pid = game["home_pitcher_id"]
        away_pid = game["away_pitcher_id"]
        home_tid = game["home_team_id"]
        away_tid = game["away_team_id"]

        # Fetch live logs
        home_p_logs = get_pitcher_recent_logs(home_pid, n=10)
        away_p_logs = get_pitcher_recent_logs(away_pid, n=10)
        home_batting = get_team_recent_batting(home_tid, n=10)
        away_batting = get_team_recent_batting(away_tid, n=10)

        # For bullpen: combine both pitchers' team logs (live path reuses same logs,
        # filtering to is_start=False within compute_bullpen_rolling)
        from data import get_pitcher_logs_season
        home_all_logs = get_pitcher_logs_season(date.today().year)
        away_all_logs = home_all_logs  # same DataFrame — both teams are in it

        # Pitcher rolling stats
        hp = compute_pitcher_rolling(home_p_logs, home_pid, today)
        ap = compute_pitcher_rolling(away_p_logs, away_pid, today)

        # Bullpen rolling stats (from season logs)
        hbp = compute_bullpen_rolling(home_all_logs, home_tid, today)
        abp = compute_bullpen_rolling(away_all_logs, away_tid, today)

        # Team offense rolling
        hoff = compute_team_offense_rolling(home_batting, home_tid, today)
        aoff = compute_team_offense_rolling(away_batting, away_tid, today)

        # Park factor
        park = PARK_FACTORS.get(str(game["home_team"]), 1.0)

        # Rest / fatigue (from season logs for full picture)
        hrest = compute_rest(home_all_logs, home_pid, home_batting, home_tid, today)
        arest = compute_rest(away_all_logs, away_pid, away_batting, away_tid, today)

        # Lineup OPS: use team rolling OPS as fallback (pre-game lineup not yet available)
        home_lineup = hoff["OPS_10"]
        away_lineup = aoff["OPS_10"]

        return {
            "home_p_ERA_3":    hp["ERA_3"],   "home_p_WHIP_3":   hp["WHIP_3"],
            "home_p_K_pct_3":  hp["K_pct_3"], "home_p_BB_pct_3": hp["BB_pct_3"],
            "home_p_K_BB_pct_3": hp["K_BB_pct_3"],
            "home_p_ERA_10":   hp["ERA_10"],  "home_p_WHIP_10":  hp["WHIP_10"],
            "home_p_K_pct_10": hp["K_pct_10"],"home_p_BB_pct_10":hp["BB_pct_10"],
            "home_p_K_BB_pct_10": hp["K_BB_pct_10"],
            "away_p_ERA_3":    ap["ERA_3"],   "away_p_WHIP_3":   ap["WHIP_3"],
            "away_p_K_pct_3":  ap["K_pct_3"], "away_p_BB_pct_3": ap["BB_pct_3"],
            "away_p_K_BB_pct_3": ap["K_BB_pct_3"],
            "away_p_ERA_10":   ap["ERA_10"],  "away_p_WHIP_10":  ap["WHIP_10"],
            "away_p_K_pct_10": ap["K_pct_10"],"away_p_BB_pct_10":ap["BB_pct_10"],
            "away_p_K_BB_pct_10": ap["K_BB_pct_10"],
            "home_bullpen_era_10":  hbp["ERA_10"], "home_bullpen_whip_10": hbp["WHIP_10"],
            "away_bullpen_era_10":  abp["ERA_10"], "away_bullpen_whip_10": abp["WHIP_10"],
            "home_team_ops_10": hoff["OPS_10"], "home_team_obp_10": hoff["OBP_10"],
            "away_team_ops_10": aoff["OPS_10"], "away_team_obp_10": aoff["OBP_10"],
            "home_lineup_ops": home_lineup, "away_lineup_ops": away_lineup,
            "park_factor": park,
            "home_pitcher_rest_days": hrest["rest_days"],
            "away_pitcher_rest_days": arest["rest_days"],
            "home_team_back_to_back": hrest["team_back_to_back"],
            "away_team_back_to_back": arest["team_back_to_back"],
        }

    except Exception as e:
        logger.error(
            "Failed to build features for %s @ %s: %s",
            game.get("away_team", "?"), game.get("home_team", "?"), e,
        )
        return None
```

- [ ] **Step 5: Run all feature tests**

```
pytest tests/test_features.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add features.py tests/test_features.py
git commit -m "feat: update build_game_features to use rolling game-log features"
```

---

## Task 6: Update `train.py` to inject new cache DataFrames

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

`get_or_build_season_features` must now load (or build) the three new cache files per season and pass them to `build_training_features`. The new signature is:
`build_training_features(historical_games, pitcher_logs, team_batting_logs, lineups)`

- [ ] **Step 1: Add failing tests to `tests/test_train.py`**

```python
# ---------------------------------------------------------------------------
# get_or_build_season_features (v2 — injects new cache DataFrames)
# ---------------------------------------------------------------------------

class TestGetOrBuildSeasonFeaturesV2:
    def _make_historical(self):
        return pd.DataFrame([{
            "game_id": 700001, "game_date": "2024-04-10", "season": 2024,
            "home_team": "NYY", "away_team": "BOS",
            "home_team_id": 147, "away_team_id": 111,
            "home_pitcher_id": 1000, "away_pitcher_id": 2000,
            "home_pitcher_name": "P1", "away_pitcher_name": "P2",
            "home_win": 1,
        }])

    def _make_pitcher_logs(self):
        import numpy as np
        rng = np.random.default_rng(0)
        rows = []
        for pid, tid in ((1000, 147), (2000, 111)):
            for i in range(5):
                rows.append({
                    "pitcher_id": pid, "team_id": tid,
                    "game_date": pd.Timestamp(f"2024-03-{i+1:02d}").date(),
                    "game_id": 600000 + i,
                    "IP": 6.0, "H": 4, "ER": 2, "BB": 2, "K": 7, "HR": 0, "BF": 24,
                    "is_start": True,
                })
        return pd.DataFrame(rows)

    def _make_batting_logs(self):
        rows = []
        for tid in (147, 111):
            for i in range(5):
                rows.append({
                    "team_id": tid,
                    "game_date": pd.Timestamp(f"2024-03-{i+1:02d}").date(),
                    "game_id": 600000 + i,
                    "R": 4, "H": 9, "BB": 3, "K": 8, "PA": 36,
                    "OBP": 0.320, "SLG": 0.420, "OPS": 0.740,
                })
        return pd.DataFrame(rows)

    def test_calls_new_data_functions_and_passes_to_build(self, tmp_path):
        import train as train_mod
        from features import FEATURE_COLUMNS

        hist = self._make_historical()
        pitcher_logs = self._make_pitcher_logs()
        batting_logs = self._make_batting_logs()
        lineups = pd.DataFrame(columns=["game_id", "team_id", "game_date", "lineup_ops"])

        with patch.object(train_mod, "CACHE_DIR", tmp_path), \
             patch("train.get_historical_game_data", return_value=hist), \
             patch("train.get_pitcher_logs_season", return_value=pitcher_logs), \
             patch("train.get_team_batting_logs_season", return_value=batting_logs), \
             patch("train.get_lineup_ops_season", return_value=lineups):
            X, y = train_mod.get_or_build_season_features(2024, force_rebuild=True)

        assert list(X.columns) == FEATURE_COLUMNS
        assert len(X) == 1

    def test_uses_feature_cache_when_available(self, tmp_path):
        import train as train_mod
        from features import FEATURE_COLUMNS
        import numpy as np

        # Pre-write a valid cache
        rng = np.random.default_rng(0)
        cache_df = pd.DataFrame(
            rng.uniform(0, 1, (5, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )
        cache_df["home_win"] = [1, 0, 1, 0, 1]
        cache_path = tmp_path / "features_2023.parquet"
        cache_df.to_parquet(cache_path, index=False)

        api_called = []
        with patch.object(train_mod, "CACHE_DIR", tmp_path), \
             patch("train.get_historical_game_data", side_effect=lambda *a: api_called.append(1) or pd.DataFrame()):
            X, y = train_mod.get_or_build_season_features(2023, force_rebuild=False)

        assert len(api_called) == 0  # cache hit, no API call
        assert len(X) == 5
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_train.py::TestGetOrBuildSeasonFeaturesV2 -v
```

Expected: FAIL — `train.get_pitcher_logs_season` not imported yet.

- [ ] **Step 3: Update imports in train.py**

Add to the existing imports at the top of `train.py`:

```python
from data import (
    get_historical_game_data,
    get_pitcher_logs_season,
    get_team_batting_logs_season,
    get_lineup_ops_season,
)
```

Replace the existing `from data import get_historical_game_data` line.

- [ ] **Step 4: Replace `get_or_build_season_features` in train.py**

Replace the entire existing `get_or_build_season_features` function (lines 45–88):

```python
def get_or_build_season_features(
    season: int,
    force_rebuild: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load season features from cache, or build from scratch if unavailable.

    Returns (X, y). Returns empty DataFrame/Series when no game data exists.

    Args:
        season: MLB season year to load or build features for.
        force_rebuild: If True, bypass any existing feature cache file.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"features_{season}.parquet"

    if not force_rebuild and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            X = df[FEATURE_COLUMNS]
            y = df["home_win"].rename("home_win")
            logger.info("Loaded cached features for %d (%d rows).", season, len(X))
            return X, y
        except Exception as e:
            logger.warning("Cache for %d is corrupt (%s) — rebuilding.", season, e)

    logger.info("Building features for %d season...", season)
    historical = get_historical_game_data([season])

    if historical.empty:
        logger.warning("No game data found for %d — skipping.", season)
        return (
            pd.DataFrame(columns=FEATURE_COLUMNS),
            pd.Series([], dtype=int, name="home_win"),
        )

    # Load or build the three new game-log caches
    pitcher_logs = get_pitcher_logs_season(season)
    team_batting_logs = get_team_batting_logs_season(season)
    lineups = get_lineup_ops_season(season, historical)

    X, y = build_training_features(historical, pitcher_logs, team_batting_logs, lineups)

    cache_df = X.copy()
    cache_df["home_win"] = y.values
    cache_df.to_parquet(cache_path, index=False)
    logger.info("Cached features for %d to %s.", season, cache_path)

    return X, y
```

- [ ] **Step 5: Run all train tests**

```
pytest tests/test_train.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "feat: update get_or_build_season_features to load and inject game-log caches"
```

---

## Task 7: Remove old data.py functions and clean up

**Files:**
- Modify: `data.py` (remove 3 old functions + dead imports)
- Modify: `tests/test_data.py` (remove imports of deleted functions)

`get_pitcher_stats`, `get_bullpen_stats`, `get_team_hitting_splits` are no longer called anywhere. `pybaseball` is no longer needed. The `_match_pitcher_row`, `_to_fg_team`, `_MLB_TO_FG_TEAM`, `_default_pitcher_stats`, `_default_rolling_pitcher_stats`, `_get_pitcher_rolling_stats` helpers are also dead code.

- [ ] **Step 1: Verify nothing imports the old functions**

```bash
grep -r "get_pitcher_stats\|get_bullpen_stats\|get_team_hitting_splits" --include="*.py" .
```

Expected: only `tests/test_data.py` (which imports them for tests) and no production code.

Also verify features.py no longer imports them:

```bash
grep "get_pitcher_stats\|get_bullpen_stats\|get_team_hitting_splits\|_get_pitcher_hand\|pybaseball" features.py
```

Expected: no matches (Task 5 replaced the imports).

- [ ] **Step 2: Remove dead test imports from `tests/test_data.py`**

Remove from the import block at the top of `tests/test_data.py`:

```python
# REMOVE these imports:
from data import (
    _safe_float,
    _match_pitcher_row,
    _to_fg_team,
    retry_on_failure,
    _default_pitcher_stats,
    _default_rolling_pitcher_stats,
)
```

Replace with just what is still needed:

```python
from data import _safe_float, retry_on_failure
```

Also remove any test classes that test deleted functions (`TestMatchPitcherRow`, `TestToFgTeam`, `TestDefaultStats`, and any tests of `get_pitcher_stats`/`get_bullpen_stats`/`get_team_hitting_splits`). Keep `TestSafeFloat` and `TestRetryDecorator`.

- [ ] **Step 3: Run test_data.py to confirm it still passes after import cleanup**

```
pytest tests/test_data.py -v
```

Expected: all PASS.

- [ ] **Step 4: Delete old functions from `data.py`**

Remove the following from `data.py`:

1. The `_MLB_TO_FG_TEAM` dict and `_to_fg_team` function (lines ~27–41)
2. The `_match_pitcher_row` function
3. The entire `get_pitcher_stats` function (including `_get_pitcher_rolling_stats`, `_default_pitcher_stats`, `_default_rolling_pitcher_stats`)
4. The entire `get_bullpen_stats` function
5. The entire `get_team_hitting_splits` function

Also remove `pybaseball` from the import block at the top of `data.py`:

```python
# REMOVE:
import pybaseball
# REMOVE:
pybaseball.cache.enable()
```

Also remove from the `from config import (...)` block:

```python
# REMOVE (no longer used):
PITCHER_ROLLING_DAYS,
BULLPEN_ROLLING_DAYS,
```

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS. No import errors.

- [ ] **Step 6: Commit**

```bash
git add data.py features.py tests/test_data.py
git commit -m "refactor: remove old FanGraphs-based data and feature functions"
```

---

## Validation (Post-Implementation)

Once all 7 tasks are complete, run a temporal holdout comparison to confirm the new model is not worse than the old:

1. Checkout the old `main` branch, run retrain on 2023–2024, save predictions on 2025 holdout
2. Checkout `feature/feature-engineering-v2`, run retrain on 2023–2024 with new features, evaluate on 2025
3. Compare: Brier score, log loss, accuracy
4. New model must match or beat old on all three metrics

This is a manual one-time comparison, not a CLI flag.
