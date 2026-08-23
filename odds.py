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
        # Totals market: over/under prices bucketed BY LINE. Books post
        # different totals for the same game (8.5 here, 9.0 there), and a price
        # is only meaningful attached to its own line. Averaging every Over
        # price together and pairing it with the average line — the old
        # behaviour — produced a quote that belonged to no real market and
        # systematically mispriced the O/U edge.
        totals_by_line: dict[float, dict[str, list[float]]] = {}

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                if key == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home_full:
                            home_probs.append(american_to_implied_prob(outcome["price"]))
                        elif outcome["name"] == away_full:
                            away_probs.append(american_to_implied_prob(outcome["price"]))
                elif key == "totals":
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point")
                        side = outcome.get("name")
                        if point is None or side not in ("Over", "Under"):
                            continue
                        bucket = totals_by_line.setdefault(
                            float(point), {"Over": [], "Under": []}
                        )
                        bucket[side].append(american_to_implied_prob(outcome["price"]))

        if not home_probs or not away_probs:
            continue

        # Consensus odds: average implied probs → convert back to American
        avg_home_prob = sum(home_probs) / len(home_probs)
        avg_away_prob = sum(away_probs) / len(away_probs)
        home_odds = implied_prob_to_american(avg_home_prob)
        away_odds = implied_prob_to_american(avg_away_prob)

        record = {
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "commence_time": event.get("commence_time", ""),
            "num_bookmakers": len(event.get("bookmakers", [])),
            "over_odds": None,
            "under_odds": None,
            "total_line": None,
        }

        # Consensus totals market: use the most widely posted line (the market
        # consensus), and average only the prices quoted at that same line.
        priced = {
            line: sides for line, sides in totals_by_line.items()
            if sides["Over"] and sides["Under"]
        }
        if priced:
            # Most books first; ties broken by the lower line for determinism.
            consensus_line = min(
                priced, key=lambda ln: (-len(priced[ln]["Over"]), ln)
            )
            sides = priced[consensus_line]
            record["over_odds"] = implied_prob_to_american(
                sum(sides["Over"]) / len(sides["Over"]))
            record["under_odds"] = implied_prob_to_american(
                sum(sides["Under"]) / len(sides["Under"]))
            record["total_line"] = consensus_line
            record["num_books_at_line"] = len(sides["Over"])

        odds_list.append(record)

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


def devig_two_way(home_odds: int, away_odds: int) -> tuple[float, float]:
    """Remove the bookmaker vig from a two-way market.

    Raw American odds imply probabilities that sum to >1 (the overround / vig).
    Normalizing each side by the total recovers the book's true (no-vig)
    probability estimate, which sums to 1.0 and is directly comparable to the
    model's probability.

    Returns (home_no_vig_prob, away_no_vig_prob).
    """
    home_implied = american_to_implied_prob(home_odds)
    away_implied = american_to_implied_prob(away_odds)
    total = home_implied + away_implied
    if total <= 0:
        return 0.5, 0.5
    return home_implied / total, away_implied / total


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


def commence_date(commence_time: str) -> Optional[str]:
    """Local (US/Eastern) calendar date for an event's UTC commence_time.

    The-Odds-API stamps commence_time in UTC, so a 7pm ET first pitch is
    23:00Z or (for late games) the following day in UTC. Bucketing by the raw
    UTC date would push every night game onto tomorrow's slate, so convert to
    Eastern — the calendar MLB schedules against — before taking the date.
    """
    if not commence_time:
        return None
    try:
        from datetime import datetime, timezone, timedelta
        ts = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # Fixed -04:00 (EDT) covers the entire MLB regular season.
        return (ts.astimezone(timezone(timedelta(hours=-4)))).date().isoformat()
    except (ValueError, TypeError):
        return None


def match_odds_to_games(odds: list[dict], games: list[dict]) -> list[dict]:
    """Match fetched odds to today's game slate.

    Odds are keyed by (home_team, away_team, commence_date). The-Odds-API
    returns every upcoming event, not just today's, and MLB teams play the same
    matchup on 3-4 consecutive days. Keying on the team pair alone (the old
    behaviour) let the LAST event for a matchup win the dict slot, so today's
    game could silently be priced off a future game in the same series — a
    different starting pitcher, a different line. Including the date makes the
    match exact, and events on other dates are ignored rather than substituted.

    Returns the games list enriched with odds data. Games without matching
    odds are excluded.
    """
    odds_lookup: dict[tuple, dict] = {}
    for o in odds:
        d = commence_date(o.get("commence_time", ""))
        key = (o["home_team"], o["away_team"], d)
        # Keep the earliest-starting event for a given matchup/date so a
        # doubleheader resolves to game 1 rather than an arbitrary one.
        prior = odds_lookup.get(key)
        if prior is None or (o.get("commence_time") or "") < (prior.get("commence_time") or ""):
            odds_lookup[key] = o

    # Fallback index for events whose commence_time is missing or unparseable.
    undated = {(o["home_team"], o["away_team"]): o
               for o in odds if commence_date(o.get("commence_time", "")) is None}

    matched = []
    for game in games:
        pair = (game["home_team"], game["away_team"])
        entry = odds_lookup.get((*pair, game["game_date"])) or undated.get(pair)
        if entry is not None:
            matched.append({**game, **entry})
        else:
            available = list(odds_lookup.keys())[:8]
            logger.warning(
                "No odds found for %s @ %s on %s — available keys (first 8): %s",
                game["away_team"], game["home_team"], game["game_date"], available,
            )

    logger.info("Matched odds for %d / %d games.", len(matched), len(games))
    return matched
