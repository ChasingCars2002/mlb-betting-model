"""Odds integration — fetch live odds from The-Odds-API and convert formats."""

import logging
from typing import Optional

import requests

from config import (
    ODDS_API_KEY,
    ODDS_API_BASE_URL,
    ODDS_SPORT,
    ODDS_REGIONS,
    ODDS_MARKETS,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
)
from data import retry_on_failure

logger = logging.getLogger(__name__)

# Common team name mappings (Odds API name → MLB abbreviation)
TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "AZ", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "ATH",
    "Athletics": "ATH", "Sacramento Athletics": "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}


@retry_on_failure
def fetch_live_odds() -> list[dict]:
    """Fetch current moneyline odds for MLB games from The-Odds-API.

    Returns list of dicts with keys: home_team, away_team, home_odds, away_odds,
    bookmaker, commence_time.
    """
    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY not set. Cannot fetch odds.")
        return []

    url = f"{ODDS_API_BASE_URL}/sports/{ODDS_SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.error("The-Odds-API request failed: %s", e)
        return []

    odds_list = []
    for event in raw:
        home_full = event.get("home_team", "")
        away_full = event.get("away_team", "")
        home_abbrev = TEAM_NAME_MAP.get(home_full, home_full)
        away_abbrev = TEAM_NAME_MAP.get(away_full, away_full)

        # Collect implied probabilities across bookmakers, then convert back.
        # Averaging American odds directly is nonlinear — go through prob space instead.
        home_probs = []
        away_probs = []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome["name"] == home_full:
                        home_probs.append(american_to_implied_prob(outcome["price"]))
                    elif outcome["name"] == away_full:
                        away_probs.append(american_to_implied_prob(outcome["price"]))

        if not home_probs or not away_probs:
            continue

        # Consensus odds: average implied probs → convert back to American
        avg_home_prob = sum(home_probs) / len(home_probs)
        avg_away_prob = sum(away_probs) / len(away_probs)
        home_odds = implied_prob_to_american(avg_home_prob)
        away_odds = implied_prob_to_american(avg_away_prob)

        odds_list.append({
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "commence_time": event.get("commence_time", ""),
            "num_bookmakers": len(event.get("bookmakers", [])),
        })

    logger.info("Fetched odds for %d games from The-Odds-API.", len(odds_list))
    return odds_list


def american_to_implied_prob(odds: int) -> float:
    """Convert American moneyline odds to implied probability.

    +150 → 100 / (150 + 100) = 0.400
    -150 → 150 / (150 + 100) = 0.600
    """
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def implied_prob_to_american(prob: float) -> int:
    """Convert an implied probability back to American moneyline odds.

    0.400 → +150
    0.600 → -150
    """
    if prob <= 0 or prob >= 1:
        raise ValueError(f"Probability must be in (0, 1), got {prob}")
    if prob < 0.5:
        # Underdog: positive odds
        return round((100.0 / prob) - 100.0)
    else:
        # Favorite: negative odds
        return round(-(prob * 100.0) / (1.0 - prob))


def decimal_to_implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability. 2.50 → 0.400."""
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal. +150 → 2.50, -150 → 1.667."""
    if odds > 0:
        return (odds / 100.0) + 1.0
    else:
        return (100.0 / abs(odds)) + 1.0


@retry_on_failure
def fetch_totals_odds() -> list[dict]:
    """Fetch current Over/Under totals odds for MLB games from The-Odds-API.

    Returns list of dicts with keys: home_team, away_team, listed_total,
    over_odds, under_odds, commence_time.
    """
    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY not set. Cannot fetch totals.")
        return []

    url = f"{ODDS_API_BASE_URL}/sports/{ODDS_SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": "totals",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.error("The-Odds-API totals request failed: %s", e)
        return []

    totals_list = []
    for event in raw:
        home_full = event.get("home_team", "")
        away_full = event.get("away_team", "")
        home_abbrev = TEAM_NAME_MAP.get(home_full, home_full)
        away_abbrev = TEAM_NAME_MAP.get(away_full, away_full)

        lines = []
        over_probs = []
        under_probs = []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "totals":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    if name == "Over":
                        lines.append(outcome.get("point", 0))
                        over_probs.append(american_to_implied_prob(outcome["price"]))
                    elif name == "Under":
                        under_probs.append(american_to_implied_prob(outcome["price"]))

        if not lines or not over_probs or not under_probs:
            continue

        avg_line = sum(lines) / len(lines)
        avg_over_prob = sum(over_probs) / len(over_probs)
        avg_under_prob = sum(under_probs) / len(under_probs)

        totals_list.append({
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "listed_total": round(avg_line, 1),
            "over_odds": implied_prob_to_american(avg_over_prob),
            "under_odds": implied_prob_to_american(avg_under_prob),
            "commence_time": event.get("commence_time", ""),
        })

    logger.info("Fetched totals for %d games from The-Odds-API.", len(totals_list))
    return totals_list


def match_totals_to_games(totals: list[dict], games: list[dict]) -> list[dict]:
    """Merge totals odds into a games list in-place.

    Games without a matching totals entry receive None values so downstream
    code can skip them cleanly.
    """
    lookup = {(t["home_team"], t["away_team"]): t for t in totals}
    for game in games:
        key = (game["home_team"], game["away_team"])
        t = lookup.get(key, {})
        game["listed_total"] = t.get("listed_total")
        game["over_odds"] = t.get("over_odds")
        game["under_odds"] = t.get("under_odds")
    return games


def match_odds_to_games(odds: list[dict], games: list[dict]) -> list[dict]:
    """Match fetched odds to today's game slate.

    Returns the games list enriched with odds data. Games without matching
    odds are excluded.
    """
    odds_lookup = {}
    for o in odds:
        key = (o["home_team"], o["away_team"])
        odds_lookup[key] = o

    matched = []
    for game in games:
        key = (game["home_team"], game["away_team"])
        if key in odds_lookup:
            game_with_odds = {**game, **odds_lookup[key]}
            matched.append(game_with_odds)
        else:
            available = list(odds_lookup.keys())[:8]
            logger.warning(
                "No odds found for %s @ %s — available keys (first 8): %s",
                game["away_team"], game["home_team"], available,
            )

    logger.info("Matched odds for %d / %d games.", len(matched), len(games))
    return matched
