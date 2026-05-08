"""Main orchestrator — daily prediction pipeline with scheduler and CLI."""

import argparse
import logging
import sys
from datetime import date

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config import (
    LOG_FILE,
    MORNING_RUN_HOUR,
    MORNING_RUN_MINUTE,
    GRADING_HOUR,
    GRADING_MINUTE,
    RETRAIN_SCHEDULE_DAY,
    RETRAIN_SCHEDULE_HOUR,
    RETRAIN_SCHEDULE_MINUTE,
)
from database import init_db, save_predictions, grade_predictions, get_roi_stats, get_all_predictions, upload_picks_to_supabase
from data import get_todays_games, get_yesterdays_results
from features import build_game_features
from model import load_model, predict_win_prob
from odds import fetch_live_odds, match_odds_to_games
from evaluate import filter_positive_ev, format_picks, format_stats, compute_confidence
from score import predict_game_scores

logger = logging.getLogger(__name__)


def post_picks_to_github_issue(picks: list[dict]) -> None:
    """Create a GitHub Issue with today's picks.

    Requires GITHUB_TOKEN and GITHUB_REPOSITORY env vars (set automatically in Actions).
    Silently skips if either is missing.
    """
    import os
    token = os.getenv("GITHUB_TOKEN", "")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not token or not repo:
        return

    today = date.today().isoformat()

    if not picks:
        title = f"⚾ MLB Picks — No picks for {today}"
        body = "No +EV picks found today."
    else:
        title = f"⚾ MLB Picks — {len(picks)} ML picks for {today}"
        ml_rows = ["## Moneyline Picks",
                   "| Game | Pick | Conf | Odds | Edge | EV | Units |",
                   "|------|------|:----:|-----:|-----:|---:|------:|"]
        for p in picks:
            matchup  = f"{p['away_team']} @ {p['home_team']}"
            odds_str = f"+{p['odds']}" if p["odds"] > 0 else str(p["odds"])
            stars    = "★" * p.get("confidence", 1)
            ml_rows.append(
                f"| {matchup} | **{p['pick']}** | {stars} | {odds_str} "
                f"| +{p['edge']:.1%} | {p['ev']:+.1%} | {p['units']:.1f}u |"
            )
        total_units = sum(p["units"] for p in picks)
        ml_rows.append(f"\n_{len(picks)} picks · {total_units:.1f} units_")
        body = "\n".join(ml_rows) + f"\n\n**Total: {len(picks)} picks · {total_units:.1f} units wagered**"

    try:
        import requests
        resp = requests.post(
            f"https://api.github.com/repos/{repo}/issues",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
            },
            json={"title": title, "body": body},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Posted picks to GitHub Issue #%d.", resp.json().get("number"))
    except Exception as e:
        logger.warning("GitHub Issue post failed: %s", e)


def post_picks_to_discord(picks: list[dict]) -> None:
    """POST today's picks to Discord via webhook.

    Silently skips if DISCORD_WEBHOOK_URL is not configured.
    Each pick is formatted as a Discord embed field.
    """
    from config import DISCORD_WEBHOOK_URL
    if not DISCORD_WEBHOOK_URL:
        return

    if not picks:
        content = "⚾ **BaseballBetBot** — No +EV picks for today."
        payload = {"content": content}
    else:
        lines = ["⚾ **BaseballBetBot — Today's Picks**\n"]
        for p in picks:
            matchup = f"{p['away_team']} @ {p['home_team']}"
            odds_str = f"+{p['odds']}" if p["odds"] > 0 else str(p["odds"])
            stars = "★" * p.get("confidence", 1)
            lines.append(
                f"🎯 **{p['pick']}** ({matchup}) {stars}\n"
                f"   Odds: `{odds_str}` · Edge: `+{p['edge']:.1%}` · "
                f"Units: `{p['units']:.1f}u` · EV: `{p['ev']:+.1%}`"
            )
        lines.append(f"\n_{len(picks)} pick(s) today — Good luck! 🍀_")
        payload = {"content": "\n".join(lines)}

    try:
        import requests
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Posted %d pick(s) to Discord.", len(picks))
    except Exception as e:
        logger.warning("Discord webhook failed: %s", e)


# ---------------------------------------------------------------------------
# Phase 1: Morning prediction run
# ---------------------------------------------------------------------------

def run_predictions(model_name: str = "xgboost"):
    """Phase 1: Fetch today's games, predict, fetch odds, calculate EV, save picks."""
    print(f"\n{'='*60}")
    print(f"  MLB BETTING MODEL — Daily Predictions ({date.today()})")
    print(f"{'='*60}")

    # Step 1: Fetch today's games
    print("\n  [1/5] Fetching today's games...")
    try:
        games = get_todays_games()
    except Exception as e:
        print(f"  ERROR: MLB Stats API unavailable — {e}")
        export_dashboard_data()
        post_picks_to_discord([])
        return
    if not games:
        print("  No games found today (off day or no probable pitchers posted). Exiting.")
        return  # Not an error — valid off day
    print(f"        Found {len(games)} games with probable pitchers.")

    # Step 2: Load model
    print(f"\n  [2/5] Loading {model_name} model...")
    try:
        model = load_model(model_name)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Models must be trained before running predictions.")
        export_dashboard_data()
        post_picks_to_discord([])
        return

    # Step 3: Build features, predict win probability, and compute score estimates
    print("\n  [3/5] Building features and predicting...")
    for game in games:
        features = build_game_features(game)
        if features is None:
            game["model_prob"] = 0.5
            game["predicted_home_runs"] = None
            game["predicted_away_runs"] = None
            logger.warning("Using default 0.5 prob for %s @ %s",
                           game["away_team"], game["home_team"])
        else:
            try:
                game["model_prob"] = predict_win_prob(model, features)
            except Exception as e:
                logger.warning("Prediction failed for %s @ %s: %s — using 0.5",
                               game["away_team"], game["home_team"], e)
                game["model_prob"] = 0.5
            try:
                scores = predict_game_scores(features)
                game["predicted_home_runs"] = scores["predicted_home_score"]
                game["predicted_away_runs"] = scores["predicted_away_score"]
            except Exception as e:
                logger.warning("Score prediction failed for %s @ %s: %s",
                               game["away_team"], game["home_team"], e)
                game["predicted_home_runs"] = None
                game["predicted_away_runs"] = None
        game["model_name"] = model_name

    # Step 4: Fetch moneyline odds and match
    print("\n  [4/5] Fetching live odds...")
    odds = fetch_live_odds()
    if not odds:
        print("  WARNING: Could not fetch odds (check ODDS_API_KEY). No picks today.")
        export_dashboard_data()
        post_picks_to_discord([])
        post_picks_to_github_issue([])
        return
    games_with_odds = match_odds_to_games(odds, games)
    if not games_with_odds:
        print("  WARNING: No games matched with odds (possible team name mapping issue). No picks today.")
        export_dashboard_data()
        post_picks_to_discord([])
        post_picks_to_github_issue([])
        return
    print(f"        Matched odds for {len(games_with_odds)} games.")

    # Step 5: Calculate EV, attach confidence, and filter picks
    print("\n  [5/5] Calculating EV and filtering picks...")
    ml_picks = filter_positive_ev(games_with_odds)
    for p in ml_picks:
        p["confidence"] = compute_confidence(p["edge"], p["ev"])
        matched = next(
            (g for g in games_with_odds
             if g["home_team"] == p["home_team"] and g["away_team"] == p["away_team"]),
            {},
        )
        p["predicted_home_runs"] = matched.get("predicted_home_runs")
        p["predicted_away_runs"] = matched.get("predicted_away_runs")

    # Display picks
    print(format_picks(ml_picks))

    # Save to database
    if ml_picks:
        save_predictions(ml_picks, bet_type="moneyline")
        print(f"  Saved {len(ml_picks)} moneyline picks to database.")
    else:
        print("  No +EV picks to save.\n")

    export_dashboard_data()
    post_picks_to_discord(ml_picks)
    post_picks_to_github_issue(ml_picks)


# ---------------------------------------------------------------------------
# Phase 2: Grade yesterday's picks
# ---------------------------------------------------------------------------

def run_grading():
    """Phase 2: Fetch yesterday's results and grade pending predictions."""
    print(f"\n{'='*60}")
    print(f"  MLB BETTING MODEL — Grading ({date.today()})")
    print(f"{'='*60}")

    print("\n  Fetching yesterday's results...")
    try:
        results = get_yesterdays_results()
    except Exception as e:
        print(f"  ERROR: MLB Stats API unavailable — {e}")
        export_dashboard_data()
        return
    if not results:
        print("  No results found for yesterday. Exiting.")
        return
    print(f"  Found results for {len(results)} games.")

    print("  Grading pending predictions...")
    grade_predictions(results)

    # Show updated stats
    show_stats()

    export_dashboard_data()


# ---------------------------------------------------------------------------
# Show lifetime stats
# ---------------------------------------------------------------------------

def show_stats():
    """Display lifetime performance statistics."""
    stats = get_roi_stats()
    output = format_stats(stats)
    print(output)


# ---------------------------------------------------------------------------
# Dashboard JSON export
# ---------------------------------------------------------------------------

def _sanitize_json(obj):
    """Recursively replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    import math
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


def export_dashboard_data():
    """Write JSON files to docs/data/ for the GitHub Pages dashboard."""
    import json
    from pathlib import Path
    from datetime import datetime

    out = Path("docs/data")
    out.mkdir(parents=True, exist_ok=True)

    ytd_since = f"{date.today().year}-01-01"
    stats = {
        "ytd": {**get_roi_stats(since=ytd_since), "since": ytd_since},
        "all_time": get_roi_stats(),
        "last_updated": datetime.utcnow().isoformat() + "Z",
    }
    (out / "stats.json").write_text(json.dumps(_sanitize_json(stats), indent=2))

    history = get_all_predictions()
    today_str = date.today().isoformat()
    # Exclude today's still-ungraded picks from history; they belong in the
    # "today's picks" section until grading completes.
    history_for_export = [
        p for p in history
        if not (p["date"] == today_str and (p.get("status") or "Pending") == "Pending")
    ]
    (out / "picks_history.json").write_text(json.dumps(_sanitize_json(history_for_export), indent=2))

    today_picks = [p for p in history if p["date"] == today_str]
    today_ml = [p for p in today_picks if p.get("bet_type", "moneyline") == "moneyline"]

    # Upload today's picks to Supabase private storage (subscriber-only).
    # If Supabase is configured, write an empty placeholder to GitHub Pages so
    # the file URL reveals nothing. Otherwise fall back to writing the real data.
    supabase_ok = upload_picks_to_supabase(today_ml, history_for_export)
    if supabase_ok:
        (out / "picks_today.json").write_text("[]")
    else:
        (out / "picks_today.json").write_text(json.dumps(_sanitize_json(today_ml), indent=2))

    if today_picks:
        top = max(today_picks, key=lambda p: p.get("edge") or 0)
        odds = top.get("odds", "")
        odds_str = f"+{odds}" if isinstance(odds, (int, float)) and odds > 0 else str(odds)
        logger.info(
            "PICK OF THE DAY: %s @ %s — Pick: %s %s (Edge: +%.1f%%)",
            top.get("away_team"), top.get("home_team"),
            top.get("pick"), odds_str,
            (top.get("edge") or 0) * 100,
        )

    logger.info("Dashboard JSON exported to %s", out)


# ---------------------------------------------------------------------------
# Incremental retrain
# ---------------------------------------------------------------------------

def run_retrain(force: bool = False, tune: bool = False):
    """Incremental model retrain entry point (used by CLI and scheduler)."""
    from train import run_incremental_retrain
    run_incremental_retrain(force=force, tune=tune)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_scheduler():
    """Run APScheduler to automate morning predictions, grading, and weekly retrain."""
    print("\n  MLB Betting Model — Scheduler Starting")
    print(f"  Predictions: Daily at {MORNING_RUN_HOUR:02d}:{MORNING_RUN_MINUTE:02d} ET")
    print(f"  Grading:     Daily at {GRADING_HOUR:02d}:{GRADING_MINUTE:02d} ET")
    print(f"  Retrain:     Weekly {RETRAIN_SCHEDULE_DAY.capitalize()} at {RETRAIN_SCHEDULE_HOUR:02d}:{RETRAIN_SCHEDULE_MINUTE:02d} ET")
    print("  Press Ctrl+C to stop.\n")

    scheduler = BlockingScheduler(timezone="US/Eastern")

    scheduler.add_job(
        run_predictions,
        CronTrigger(hour=MORNING_RUN_HOUR, minute=MORNING_RUN_MINUTE),
        id="daily_predictions",
        name="Daily MLB Predictions",
    )

    scheduler.add_job(
        run_grading,
        CronTrigger(hour=GRADING_HOUR, minute=GRADING_MINUTE),
        id="daily_grading",
        name="Daily Prediction Grading",
    )

    scheduler.add_job(
        run_retrain,
        CronTrigger(
            day_of_week=RETRAIN_SCHEDULE_DAY,
            hour=RETRAIN_SCHEDULE_HOUR,
            minute=RETRAIN_SCHEDULE_MINUTE,
        ),
        id="weekly_retrain",
        name="Weekly Incremental Retrain",
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n  Scheduler stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLB Betting Model")
    parser.add_argument(
        "--run-now", action="store_true",
        help="Run today's predictions immediately.",
    )
    parser.add_argument(
        "--grade", action="store_true",
        help="Grade yesterday's pending predictions.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show lifetime performance stats.",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train models (equivalent to running train.py).",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Incremental retrain: use cached features for completed seasons, rebuild current season only.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="With --retrain: wipe all caches and do a full rebuild from scratch.",
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["xgboost", "logistic_regression", "lightgbm"],
        help="Which model to use for predictions (default: xgboost).",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="With --train/--retrain: run Optuna hyperparameter tuning first (~5-10 min).",
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Start the daily scheduler.",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export JSON files to docs/data/ for the GitHub Pages dashboard.",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(LOG_FILE)),
        ],
    )

    # Initialize database
    init_db()

    # Guard: --force is only meaningful with --retrain
    if args.force and not args.retrain:
        parser.error("--force requires --retrain")

    # Determine action
    if args.train:
        from train import run_incremental_retrain
        run_incremental_retrain(force=True, tune=args.tune)
    elif args.retrain:
        run_retrain(force=args.force, tune=args.tune)
    elif args.run_now:
        run_predictions(model_name=args.model)
    elif args.grade:
        run_grading()
    elif args.stats:
        show_stats()
    elif args.schedule:
        run_scheduler()
    elif args.export:
        export_dashboard_data()
    else:
        # Default: run predictions
        parser.print_help()
        print("\nExamples:")
        print("  python main.py --train               # Full retrain (wipes cache)")
        print("  python main.py --retrain             # Incremental retrain (uses cache)")
        print("  python main.py --retrain --force     # Force full rebuild via --retrain")
        print("  python main.py --run-now             # Run today's predictions")
        print("  python main.py --grade               # Grade yesterday's picks")
        print("  python main.py --stats               # Show lifetime stats")
        print("  python main.py --schedule            # Start daily scheduler")

if __name__ == "__main__":
    main()