"""Data acquisition module — fetches MLB game data, pitcher stats, and results."""

import logging
import time
import functools
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import pybaseball

from config import (
    MLB_STATS_API_BASE,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    PITCHER_ROLLING_DAYS,
    BULLPEN_ROLLING_DAYS,
)

logger = logging.getLogger(__name__)

# Suppress pybaseball's verbose output
pybaseball.cache.enable()

# FanGraphs uses different abbreviations for some MLB teams
_MLB_TO_FG_TEAM = {
    "KC": "KCR",
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "WSH": "WSN",
    "CLE": "CLE",  # Guardians kept abbreviation
    "AZ": "ARI",   # MLB Stats API now uses AZ; FanGraphs still uses ARI
    "ATH": "OAK",  # Athletics relocated; FanGraphs still uses OAK
}


def _to_fg_team(mlb_abbrev: str) -> str:
    """Convert an MLB Stats API team abbreviation to its FanGraphs equivalent."""
    return _MLB_TO_FG_TEAM.get(mlb_abbrev, mlb_abbrev)


# 5-year park run factors (1.00 = league average, >1.00 = hitter-friendly)
# Source: Baseball Reference 2023-2025 multi-year park factors
PARK_FACTORS = {
    "ARI": 1.03, "AZ": 1.03,  # both abbreviations used for Diamondbacks
    "ATL": 0.99, "BAL": 1.01, "BOS": 1.03, "CHC": 1.02,
    "CWS": 0.98, "CIN": 1.04, "CLE": 0.97, "COL": 1.22, "DET": 0.98,
    "HOU": 0.98, "KC":  1.01, "LAA": 0.99, "LAD": 0.97, "MIA": 0.96,
    "MIL": 1.00, "MIN": 1.01, "NYM": 0.99, "NYY": 1.01, "ATH": 0.97,
    "PHI": 1.02, "PIT": 0.99, "SD":  0.97, "SEA": 0.97, "SF":  0.95,
    "STL": 0.99, "TB":  0.98, "TEX": 1.02, "TOR": 1.01, "WSH": 1.00,
}

def get_park_factor(team_abbrev: str) -> float:
    """Return the park run factor for the home team's stadium. Default 1.0."""
    return PARK_FACTORS.get(team_abbrev, 1.0)


def get_team_rolling_ops(team_id: int, game_date_str: str, n_games: int = 10) -> float:
    """Fetch a team's OPS over their last n_games before game_date.

    Uses MLB Stats API team game logs. Returns league-average OPS (0.720)
    on API failure or insufficient data.

    Args:
        team_id: MLB team ID (integer).
        game_date_str: Game date as YYYY-MM-DD string.
        n_games: Number of recent games to average (default 10).
    """
    DEFAULT_OPS = 0.720
    try:
        game_date = date.fromisoformat(game_date_str)
        season = game_date.year
        data = _mlb_api_get(
            f"teams/{team_id}/stats",
            params={
                "stats": "gameLog",
                "group": "hitting",
                "season": season,
                "gameType": "R",
            },
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        # Keep only games before game_date
        past = [
            s for s in splits
            if s.get("date") and s["date"] < game_date_str
        ]
        if len(past) < 3:  # Need at least 3 games
            return DEFAULT_OPS
        recent = sorted(past, key=lambda s: s["date"])[-n_games:]
        # Simpler: use OPS directly from each split's stat block if available
        ops_values = []
        for g in recent:
            stat = g.get("stat", {})
            ops_str = stat.get("ops", "")
            try:
                ops_values.append(float(ops_str))
            except (ValueError, TypeError):
                pass
        if not ops_values:
            return DEFAULT_OPS
        return round(sum(ops_values) / len(ops_values), 4)
    except Exception as e:
        logger.debug("Could not get rolling OPS for team %d: %s", team_id, e)
        return DEFAULT_OPS


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_on_failure(func):
    """Retry a function up to MAX_RETRIES times with exponential backoff."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "%s attempt %d/%d failed: %s. Retrying in %ds...",
                    func.__name__, attempt, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
        logger.error("%s failed after %d attempts.", func.__name__, MAX_RETRIES)
        raise last_exc
    return wrapper


# ---------------------------------------------------------------------------
# MLB Stats API helpers
# ---------------------------------------------------------------------------

@retry_on_failure
def _mlb_api_get(endpoint: str, params: Optional[dict] = None) -> dict:
    """Make a GET request to the MLB Stats API."""
    url = f"{MLB_STATS_API_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Today's games & probable pitchers
# ---------------------------------------------------------------------------

@retry_on_failure
def get_todays_games(target_date: Optional[date] = None) -> list[dict]:
    """Fetch today's MLB games with probable starting pitchers.

    Returns list of dicts with keys: game_id, home_team, away_team,
    home_pitcher_id, home_pitcher_name, away_pitcher_id, away_pitcher_name,
    home_pitcher_hand, away_pitcher_hand, game_date.
    """
    d = target_date or date.today()
    date_str = d.strftime("%Y-%m-%d")

    data = _mlb_api_get(
        "schedule",
        params={
            "sportId": 1,
            "date": date_str,
            "hydrate": "probablePitcher,team",
        },
    )

    games = []
    for game_date in data.get("dates", []):
        for g in game_date.get("games", []):
            if g.get("status", {}).get("abstractGameCode") == "F":
                continue  # skip completed games

            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})

            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})

            if not home_pitcher.get("id") or not away_pitcher.get("id"):
                logger.warning(
                    "Missing probable pitcher for %s vs %s, skipping.",
                    away.get("team", {}).get("abbreviation", "?"),
                    home.get("team", {}).get("abbreviation", "?"),
                )
                continue

            games.append({
                "game_id": g["gamePk"],
                "game_date": date_str,
                "home_team": home["team"]["abbreviation"],
                "away_team": away["team"]["abbreviation"],
                "home_team_id": home["team"]["id"],
                "away_team_id": away["team"]["id"],
                "home_pitcher_id": home_pitcher["id"],
                "home_pitcher_name": home_pitcher.get("fullName", "Unknown"),
                "away_pitcher_id": away_pitcher["id"],
                "away_pitcher_name": away_pitcher.get("fullName", "Unknown"),
                "home_pitcher_hand": _get_pitcher_hand(home_pitcher["id"]),
                "away_pitcher_hand": _get_pitcher_hand(away_pitcher["id"]),
            })

    logger.info("Found %d games with probable pitchers for %s.", len(games), date_str)
    return games


_pitcher_hand_cache: dict[int, str] = {}


def _get_pitcher_hand(pitcher_id: int) -> str:
    """Get a pitcher's throwing hand (L or R) from MLB Stats API."""
    if pitcher_id in _pitcher_hand_cache:
        return _pitcher_hand_cache[pitcher_id]
    try:
        data = _mlb_api_get(f"people/{pitcher_id}")
        hand = data["people"][0]["pitchHand"]["code"]
        _pitcher_hand_cache[pitcher_id] = hand
        return hand
    except Exception:
        logger.warning("Could not determine hand for pitcher %d, defaulting to R.", pitcher_id)
        return "R"


def get_pitcher_days_rest(pitcher_id: int, game_date_str: str) -> float:
    """Return days since pitcher's last appearance before game_date.

    Queries MLB Stats API for the pitcher's recent game log.
    Returns days since last start (capped at 10 for "fresh" pitchers).
    Returns 5.0 (league-average rest) on API failure or no data.

    Args:
        pitcher_id: MLB player ID.
        game_date_str: Game date as YYYY-MM-DD string.
    """
    DEFAULT_REST = 5.0
    try:
        game_date = date.fromisoformat(game_date_str)
        season = game_date.year
        data = _mlb_api_get(
            f"people/{pitcher_id}/stats",
            params={
                "stats": "gameLog",
                "group": "pitching",
                "season": season,
                "gameType": "R",
            },
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        # Filter to starts (inningsPitched > 0) before game_date
        past_games = [
            s for s in splits
            if s.get("date") and s["date"] < game_date_str
        ]
        if not past_games:
            return DEFAULT_REST
        # Most recent game
        last_game_date = date.fromisoformat(
            max(past_games, key=lambda s: s["date"])["date"]
        )
        days = (game_date - last_game_date).days
        return float(min(days, 10))  # cap at 10
    except Exception as e:
        logger.debug("Could not get days rest for pitcher %d: %s", pitcher_id, e)
        return DEFAULT_REST


# ---------------------------------------------------------------------------
# Pitcher stats via pybaseball (FanGraphs / Statcast)
# ---------------------------------------------------------------------------

def _match_pitcher_row(stats: pd.DataFrame, pitcher_name: str) -> Optional[pd.Series]:
    """Match a pitcher in a FanGraphs DataFrame.

    Tries exact full-name match first; falls back to last-name substring.
    Returns None if no match found.
    """
    # 1. Exact full name (case-insensitive)
    mask_full = stats["Name"].str.lower() == pitcher_name.lower()
    if mask_full.sum() == 1:
        return stats[mask_full].iloc[0]
    if mask_full.sum() > 1:
        logger.warning("Multiple exact matches for '%s', using first.", pitcher_name)
        return stats[mask_full].iloc[0]

    # 2. Last-name substring fallback
    if not pitcher_name or not pitcher_name.strip():
        return None
    last_name = pitcher_name.split()[-1]
    mask_last = stats["Name"].str.contains(last_name, case=False, na=False)
    if mask_last.sum() == 0:
        return None
    if mask_last.sum() > 1:
        logger.warning(
            "Ambiguous last-name match for '%s' (%d hits), using first.",
            pitcher_name, mask_last.sum(),
        )
    return stats[mask_last].iloc[0]


@retry_on_failure
def get_pitcher_stats(
    pitcher_name: str,
    season: Optional[int] = None,
    rolling_days: int = PITCHER_ROLLING_DAYS,
    use_rolling: bool = True,
) -> dict:
    """Get pitcher sabermetric stats: xFIP, SIERA, K-BB%, WHIP.

    Returns both rolling-window and season-long stats.

    Args:
        use_rolling: When True (prediction path), fetch a real rolling window
            via pitching_stats_range. When False (training path), reuse season
            stats for the rolling slots — historical rolling windows can't be
            reconstructed from today's FanGraphs data.
    """
    season = season or date.today().year

    try:
        stats = pybaseball.pitching_stats(season, season, qual=1)
    except Exception as e:
        logger.warning("pybaseball pitching_stats failed: %s. Using defaults.", e)
        return _default_pitcher_stats()

    row = _match_pitcher_row(stats, pitcher_name)
    if row is None:
        logger.warning("Pitcher '%s' not found in %d stats. Using defaults.", pitcher_name, season)
        return _default_pitcher_stats()

    season_stats = {
        "xFIP_season": _safe_float(row.get("xFIP")),
        "SIERA_season": _safe_float(row.get("SIERA")),
        "K_BB_pct_season": _safe_float(row.get("K-BB%")),
        "WHIP_season": _safe_float(row.get("WHIP")),
    }

    if use_rolling:
        rolling_stats = _get_pitcher_rolling_stats(pitcher_name, rolling_days)
    else:
        # Training path: mirror season stats into rolling slots
        rolling_stats = {
            "xFIP_rolling": season_stats["xFIP_season"],
            "SIERA_rolling": season_stats["SIERA_season"],
            "K_BB_pct_rolling": season_stats["K_BB_pct_season"],
            "WHIP_rolling": season_stats["WHIP_season"],
        }

    return {**season_stats, **rolling_stats}


def _get_pitcher_rolling_stats(pitcher_name: str, days: int) -> dict:
    """Compute rolling stats from FanGraphs date-range data (actual window, not season proxy)."""
    end = date.today()
    start = end - timedelta(days=days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    try:
        stats = pybaseball.pitching_stats_range(start_str, end_str)
        if stats is None or stats.empty:
            return _default_rolling_pitcher_stats()

        row = _match_pitcher_row(stats, pitcher_name)
        if row is None:
            logger.warning(
                "No rolling stats for '%s' in last %d days. Using season stats as fallback.",
                pitcher_name, days,
            )
            return _default_rolling_pitcher_stats()

        return {
            "xFIP_rolling": _safe_float(row.get("xFIP")),
            "SIERA_rolling": _safe_float(row.get("SIERA")),
            "K_BB_pct_rolling": _safe_float(row.get("K-BB%")),
            "WHIP_rolling": _safe_float(row.get("WHIP")),
        }
    except IndexError:
        logger.debug(
            "No FanGraphs data yet for '%s' in last %d days (early season). Using defaults.",
            pitcher_name, days,
        )
        return _default_rolling_pitcher_stats()
    except Exception as e:
        logger.warning("Rolling stats fetch failed for %s: %s", pitcher_name, e)
        return _default_rolling_pitcher_stats()


def _default_pitcher_stats() -> dict:
    return {
        "xFIP_season": 4.20, "SIERA_season": 4.20,
        "K_BB_pct_season": 10.0, "WHIP_season": 1.30,
        "xFIP_rolling": 4.20, "SIERA_rolling": 4.20,
        "K_BB_pct_rolling": 10.0, "WHIP_rolling": 1.30,
    }


def _default_rolling_pitcher_stats() -> dict:
    return {
        "xFIP_rolling": 4.20, "SIERA_rolling": 4.20,
        "K_BB_pct_rolling": 10.0, "WHIP_rolling": 1.30,
    }


# ---------------------------------------------------------------------------
# Bullpen stats
# ---------------------------------------------------------------------------

@retry_on_failure
def get_bullpen_stats(
    team_abbrev: str,
    season: Optional[int] = None,
    rolling_days: int = BULLPEN_ROLLING_DAYS,
) -> dict:
    """Get team bullpen aggregate ERA and FIP.

    Returns: {"bullpen_era": float, "bullpen_fip": float}
    """
    season = season or date.today().year
    fg_team = _to_fg_team(team_abbrev)

    try:
        pitching = pybaseball.pitching_stats(season, season, qual=1)
        # Filter to relievers. Both conditions must hold:
        #   - GS < 5: pitcher does not have a primary starting role
        #   - G - GS > 10: pitcher has substantial relief workload
        # Using OR here would admit swingmen (e.g. 22 GS, 30 G) into the
        # bullpen aggregate, diluting it with rotation-quality innings.
        if "GS" in pitching.columns and "G" in pitching.columns:
            relievers = pitching[
                (pitching["GS"] < 5) & (pitching["G"] - pitching["GS"] > 10)
            ]
            # Exact team match with FanGraphs abbreviation; fall back to contains
            team_relievers = relievers[relievers["Team"] == fg_team]
            if team_relievers.empty:
                team_relievers = relievers[
                    relievers["Team"].str.contains(team_abbrev, case=False, na=False)
                ]
            if not team_relievers.empty:
                return {
                    "bullpen_era": round(team_relievers["ERA"].mean(), 3),
                    "bullpen_fip": round(team_relievers["FIP"].mean(), 3),
                }
    except Exception as e:
        logger.warning("Bullpen stats failed for %s: %s. Using defaults.", team_abbrev, e)

    return {"bullpen_era": 4.00, "bullpen_fip": 4.00}


# ---------------------------------------------------------------------------
# Team hitting splits (vs L/R)
# ---------------------------------------------------------------------------

@retry_on_failure
def get_team_hitting_splits(
    team_abbrev: str,
    vs_hand: str,
    season: Optional[int] = None,
) -> dict:
    """Get team wRC+ and OPS against a specific pitcher handedness.

    Args:
        team_abbrev: Team abbreviation (e.g., "NYY").
        vs_hand: "L" or "R" for opposing pitcher's throwing hand.
        season: MLB season year.

    Returns: {"wrc_plus": float, "ops": float}
    """
    season = season or date.today().year
    fg_team = _to_fg_team(team_abbrev)

    try:
        batting = pybaseball.batting_stats(season, season, qual=1)
        # Exact team match with FanGraphs abbreviation; fall back to contains
        team_hitters = batting[batting["Team"] == fg_team]
        if team_hitters.empty:
            team_hitters = batting[
                batting["Team"].str.contains(team_abbrev, case=False, na=False)
            ]
        if not team_hitters.empty:
            wrc_plus = team_hitters["wRC+"].mean() if "wRC+" in team_hitters.columns else 100.0
            ops = team_hitters["OPS"].mean() if "OPS" in team_hitters.columns else 0.740

            # TODO: This is a placeholder. The function signature accepts a
            # pitcher hand, but FanGraphs season-aggregate batting stats are
            # not split by opposing pitcher handedness. Applying a uniform
            # ±3% adjustment to every team erases all team-level platoon
            # signal (lefty-heavy lineups that mash RHP look the same as
            # righty-heavy ones). Replace with vs-L / vs-R splits from
            # pybaseball.team_batting_bref_split or the FanGraphs splits API.
            if vs_hand == "L":
                wrc_plus *= 0.97
                ops *= 0.97
            else:
                wrc_plus *= 1.03
                ops *= 1.03

            return {
                "wrc_plus": round(wrc_plus, 1),
                "ops": round(ops, 3),
            }
    except Exception as e:
        logger.warning("Hitting splits failed for %s vs %s: %s", team_abbrev, vs_hand, e)

    return {"wrc_plus": 100.0, "ops": 0.740}


# ---------------------------------------------------------------------------
# Historical data for model training
# ---------------------------------------------------------------------------

@retry_on_failure
def get_historical_game_data(seasons: list[int]) -> pd.DataFrame:
    """Pull historical game-level data for training the model.

    Returns DataFrame with one row per game, columns:
    game_date, home_team, away_team, home_score, away_score, home_win,
    home_pitcher_name, away_pitcher_name, home_pitcher_id, away_pitcher_id,
    home_pitcher_hand, away_pitcher_hand, season.
    """
    all_games = []

    for season in seasons:
        logger.info("Fetching %d schedule...", season)
        try:
            data = _mlb_api_get(
                "schedule",
                params={
                    "sportId": 1,
                    "season": season,
                    "gameType": "R",  # regular season
                    "hydrate": "probablePitcher,linescore,team",
                },
            )

            for game_date in data.get("dates", []):
                for g in game_date.get("games", []):
                    if g.get("status", {}).get("codedGameState") != "F":
                        continue  # only final games

                    home = g.get("teams", {}).get("home", {})
                    away = g.get("teams", {}).get("away", {})
                    home_score = home.get("score")
                    away_score = away.get("score")

                    if home_score is None or away_score is None:
                        continue

                    home_pitcher = home.get("probablePitcher", {})
                    away_pitcher = away.get("probablePitcher", {})

                    all_games.append({
                        "game_date": game_date["date"],
                        "season": season,
                        "game_id": g["gamePk"],
                        "home_team": home["team"]["abbreviation"],
                        "away_team": away["team"]["abbreviation"],
                        "home_team_id": home["team"]["id"],
                        "away_team_id": away["team"]["id"],
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_win": 1 if home_score > away_score else 0,
                        "home_pitcher_id": home_pitcher.get("id"),
                        "home_pitcher_name": home_pitcher.get("fullName", "Unknown"),
                        "away_pitcher_id": away_pitcher.get("id"),
                        "away_pitcher_name": away_pitcher.get("fullName", "Unknown"),
                    })

        except Exception as e:
            logger.error("Failed to fetch %d data: %s", season, e)
            continue

    df = pd.DataFrame(all_games)
    logger.info("Loaded %d historical games across seasons %s.", len(df), seasons)
    return df


# ---------------------------------------------------------------------------
# Yesterday's results for grading
# ---------------------------------------------------------------------------

@retry_on_failure
def get_yesterdays_results(target_date: Optional[date] = None) -> dict[str, dict]:
    """Fetch final scores for yesterday's games.

    Returns dict keyed by "AWAY @ HOME" with values:
    {"home_score": int, "away_score": int, "winner": str (team abbrev)}.
    """
    d = target_date or (date.today() - timedelta(days=1))
    date_str = d.strftime("%Y-%m-%d")

    data = _mlb_api_get(
        "schedule",
        params={
            "sportId": 1,
            "date": date_str,
            "hydrate": "linescore,team",
        },
    )

    results = {}
    for game_date in data.get("dates", []):
        for g in game_date.get("games", []):
            if g.get("status", {}).get("codedGameState") != "F":
                continue

            home = g["teams"]["home"]
            away = g["teams"]["away"]
            home_abbrev = home["team"]["abbreviation"]
            away_abbrev = away["team"]["abbreviation"]
            home_score = home.get("score", 0)
            away_score = away.get("score", 0)

            key = f"{away_abbrev} @ {home_abbrev}"
            results[key] = {
                "home_score": home_score,
                "away_score": away_score,
                "winner": home_abbrev if home_score > away_score else away_abbrev,
            }

    logger.info("Fetched %d game results for %s.", len(results), date_str)
    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if val is None:
        return default
    try:
        if isinstance(val, str):
            val = val.replace("%", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return default
