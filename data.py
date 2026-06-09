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


# ---------------------------------------------------------------------------
# FanGraphs User-Agent fix
# ---------------------------------------------------------------------------
# pybaseball scrapes FanGraphs with the default ``python-requests`` User-Agent,
# which FanGraphs blocks with HTTP 403. That silently degraded every pitching,
# bullpen, and hitting feature to league-average defaults (see the flood of
# "Received status code 403 ... Using defaults" warnings in the logs). We inject
# a real browser User-Agent on every outbound request so the scrape succeeds
# from CI/cloud IPs. requests.get() and Session.get() both route through
# Session.request, so patching it once covers all of pybaseball; adding a UA is
# harmless for the JSON APIs (MLB Stats, The-Odds-API, Supabase) we also call.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _install_browser_user_agent() -> None:
    """Make every requests call send a browser User-Agent (FanGraphs 403 fix)."""
    session_cls = requests.sessions.Session
    if getattr(session_cls, "_mlb_ua_patched", False):
        return
    _orig_request = session_cls.request

    @functools.wraps(_orig_request)
    def _request_with_ua(self, method, url, *args, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        if not any(k.lower() == "user-agent" for k in headers):
            headers["User-Agent"] = _BROWSER_USER_AGENT
            kwargs["headers"] = headers
        return _orig_request(self, method, url, *args, **kwargs)

    session_cls.request = _request_with_ua
    session_cls._mlb_ua_patched = True


_install_browser_user_agent()


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


# ---------------------------------------------------------------------------
# MLB Stats API metric helpers (replaces the FanGraphs/pybaseball feed, which
# returns HTTP 403 to GitHub runner IPs). FanGraphs-specific metrics (xFIP,
# SIERA, wRC+) aren't exposed by MLB, so we map to the closest real signals:
#   xFIP slot  <- FIP (computed)     SIERA slot <- ERA
#   K_BB_pct   <- (K - BB) / BF      WHIP       <- whip
#   wRC+ slot  <- OPS-scaled index
# Absolute scale doesn't matter because the model is retrained on these values.
# ---------------------------------------------------------------------------

_FIP_CONSTANT = 3.10  # nominal; only relative differences matter after retrain

# season -> {player_id: stat_dict}; (team_id, season[, hand]) -> metrics
_pitching_leaderboard_cache: dict[int, dict] = {}
_team_pitching_cache: dict[tuple, dict] = {}
_team_hitting_cache: dict[tuple, dict] = {}
_team_id_cache: dict[str, int] = {}


def _ip_to_float(ip) -> float:
    """Convert MLB 'inningsPitched' (e.g. '12.1' = 12 and 1/3) to a float."""
    try:
        s = str(ip)
        if "." in s:
            whole, frac = s.split(".")
            return int(whole) + {"0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(frac, 0.0)
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _compute_fip(stat: dict) -> float:
    """FIP from a raw MLB pitching stat line: (13HR + 3(BB+HBP) - 2K)/IP + C."""
    ip = _ip_to_float(stat.get("inningsPitched"))
    if ip <= 0:
        return 4.20
    hr = _safe_float(stat.get("homeRuns"))
    bb = _safe_float(stat.get("baseOnBalls"))
    hbp = _safe_float(stat.get("hitByPitch"))
    k = _safe_float(stat.get("strikeOuts"))
    return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + _FIP_CONSTANT, 3)


def _pitcher_metrics(stat: dict) -> dict:
    """Map an MLB pitching stat line to (xFIP-slot, SIERA-slot, K-BB%, WHIP)."""
    bf = _safe_float(stat.get("battersFaced"))
    k = _safe_float(stat.get("strikeOuts"))
    bb = _safe_float(stat.get("baseOnBalls"))
    k_bb_pct = round((k - bb) / bf * 100, 2) if bf > 0 else 10.0
    return {
        "xFIP": _compute_fip(stat),
        "SIERA": _safe_float(stat.get("era"), default=4.20),
        "K_BB_pct": k_bb_pct,
        "WHIP": _safe_float(stat.get("whip"), default=1.30),
    }


@retry_on_failure
def _pitching_leaderboard(season: int) -> dict:
    """All pitchers' season stats for `season`, keyed by player id (cached)."""
    if season in _pitching_leaderboard_cache:
        return _pitching_leaderboard_cache[season]
    data = _mlb_api_get("stats", params={
        "stats": "season", "group": "pitching", "season": season,
        "sportId": 1, "playerPool": "all", "limit": 3000,
    })
    board = {}
    for s in (data.get("stats") or []):
        for sp in s.get("splits", []):
            pid = sp.get("player", {}).get("id")
            if pid is not None:
                board[pid] = sp.get("stat", {})
    _pitching_leaderboard_cache[season] = board
    logger.info("Loaded MLB pitching leaderboard for %d: %d pitchers.", season, len(board))
    return board


@retry_on_failure
def _team_id_from_abbrev(team_abbrev) -> Optional[int]:
    """Resolve an MLB team abbreviation to its numeric id (cached)."""
    if isinstance(team_abbrev, int):
        return team_abbrev
    if str(team_abbrev).isdigit():
        return int(team_abbrev)
    if not _team_id_cache:
        data = _mlb_api_get("teams", params={"sportId": 1})
        for t in data.get("teams", []):
            if t.get("abbreviation"):
                _team_id_cache[t["abbreviation"]] = t["id"]
    return _team_id_cache.get(team_abbrev)


def get_pitcher_stats(
    pitcher_id=None,
    pitcher_name: str = "",
    season: Optional[int] = None,
    rolling_days: int = PITCHER_ROLLING_DAYS,
    use_rolling: bool = True,
) -> dict:
    """Get pitcher metrics (xFIP/SIERA/K-BB%/WHIP slots) from the MLB Stats API.

    Looks the pitcher up by id in the season leaderboard. `pitcher_name` is kept
    for logging/back-compat only.

    Args:
        use_rolling: When True (prediction path), fetch a real recent-form window
            via the byDateRange endpoint. When False (training path), reuse season
            stats for the rolling slots — historical rolling windows aren't
            reconstructable cheaply.
    """
    season = season or date.today().year

    if pitcher_id is None or (isinstance(pitcher_id, float) and pitcher_id != pitcher_id):
        return _default_pitcher_stats()
    try:
        pitcher_id = int(pitcher_id)
    except (TypeError, ValueError):
        return _default_pitcher_stats()

    try:
        board = _pitching_leaderboard(season)
    except Exception as e:
        logger.warning("MLB pitching leaderboard failed for %d: %s. Using defaults.", season, e)
        return _default_pitcher_stats()

    stat = board.get(pitcher_id)
    if not stat:
        logger.warning("Pitcher id=%s (%s) not in %d leaderboard. Using defaults.",
                       pitcher_id, pitcher_name, season)
        return _default_pitcher_stats()

    m = _pitcher_metrics(stat)
    season_stats = {
        "xFIP_season": m["xFIP"], "SIERA_season": m["SIERA"],
        "K_BB_pct_season": m["K_BB_pct"], "WHIP_season": m["WHIP"],
    }

    if use_rolling:
        rolling_stats = _get_pitcher_rolling_stats(pitcher_id, rolling_days, season)
    else:
        # Training path: mirror season stats into rolling slots
        rolling_stats = {
            "xFIP_rolling": season_stats["xFIP_season"],
            "SIERA_rolling": season_stats["SIERA_season"],
            "K_BB_pct_rolling": season_stats["K_BB_pct_season"],
            "WHIP_rolling": season_stats["WHIP_season"],
        }

    return {**season_stats, **rolling_stats}


def _get_pitcher_rolling_stats(pitcher_id: int, days: int, season: int) -> dict:
    """Recent-form pitcher metrics via the MLB byDateRange endpoint."""
    end = date.today()
    start = end - timedelta(days=days)
    try:
        data = _mlb_api_get(f"people/{pitcher_id}/stats", params={
            "stats": "byDateRange", "group": "pitching",
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": end.strftime("%Y-%m-%d"),
            "sportId": 1, "season": season,
        })
        splits = (data.get("stats") or [{}])[0].get("splits", [])
        if not splits:
            return _default_rolling_pitcher_stats()
        m = _pitcher_metrics(splits[0].get("stat", {}))
        return {
            "xFIP_rolling": m["xFIP"], "SIERA_rolling": m["SIERA"],
            "K_BB_pct_rolling": m["K_BB_pct"], "WHIP_rolling": m["WHIP"],
        }
    except Exception as e:
        logger.warning("Rolling stats fetch failed for pitcher %s: %s", pitcher_id, e)
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
    """Team pitching-staff ERA and FIP from the MLB Stats API.

    NOTE: MLB's simple team-stats endpoint doesn't split starters from relievers,
    so this uses the team's overall pitching as a proxy for staff/bullpen quality.

    Returns: {"bullpen_era": float, "bullpen_fip": float}
    """
    season = season or date.today().year
    key = (team_abbrev, season)
    if key in _team_pitching_cache:
        return _team_pitching_cache[key]

    result = {"bullpen_era": 4.00, "bullpen_fip": 4.00}
    try:
        team_id = _team_id_from_abbrev(team_abbrev)
        if team_id:
            data = _mlb_api_get(f"teams/{team_id}/stats", params={
                "stats": "season", "group": "pitching", "season": season, "sportId": 1,
            })
            splits = (data.get("stats") or [{}])[0].get("splits", [])
            if splits:
                stat = splits[0].get("stat", {})
                result = {
                    "bullpen_era": _safe_float(stat.get("era"), default=4.00),
                    "bullpen_fip": _compute_fip(stat),
                }
    except Exception as e:
        logger.warning("Team pitching stats failed for %s: %s. Using defaults.", team_abbrev, e)

    _team_pitching_cache[key] = result
    return result


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
    key = (team_abbrev, vs_hand, season)
    if key in _team_hitting_cache:
        return _team_hitting_cache[key]

    result = {"wrc_plus": 100.0, "ops": 0.740}
    try:
        team_id = _team_id_from_abbrev(team_abbrev)
        if team_id:
            # Real platoon split: vs LHP (vl) / vs RHP (vr).
            sit = "vl" if vs_hand == "L" else "vr"
            data = _mlb_api_get(f"teams/{team_id}/stats", params={
                "stats": "statSplits", "group": "hitting", "season": season,
                "sportId": 1, "sitCodes": sit,
            })
            ops = None
            for s in (data.get("stats") or []):
                for sp in s.get("splits", []):
                    val = sp.get("stat", {}).get("ops")
                    if val is not None:
                        ops = _safe_float(val)
            # Fallback to overall season hitting if the split is unavailable.
            if ops is None:
                data = _mlb_api_get(f"teams/{team_id}/stats", params={
                    "stats": "season", "group": "hitting", "season": season, "sportId": 1,
                })
                splits = (data.get("stats") or [{}])[0].get("splits", [])
                if splits:
                    ops = _safe_float(splits[0].get("stat", {}).get("ops"), default=0.740)
            if ops:
                # No wRC+ in MLB's API; approximate from OPS (league avg OPS ~.720 -> ~100).
                result = {"ops": round(ops, 3), "wrc_plus": round(100.0 * ops / 0.720, 1)}
    except Exception as e:
        logger.warning("Hitting splits failed for %s vs %s: %s", team_abbrev, vs_hand, e)

    _team_hitting_cache[key] = result
    return result


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
