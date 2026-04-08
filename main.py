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
from database import init_db, save_predictions, grade_predictions, get_roi_stats
from data import get_todays_games, get_yesterdays_results
from features import build_game_features
from model import load_model, predict_win_prob
from odds import fetch_live_odds, match_odds_to_games
from evaluate import filter_positive_ev, format_picks, format_stats

logger = logging.getLogger(__name__)


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
    games = get_todays_games()
    if not games:
        print("  No games found today. Exiting.")
        return
    print(f"        Found {len(games)} games with probable pitchers.")

    # Step 2: Load model
    print(f"\n  [2/5] Loading {model_name} model...")
    try:
        model = load_model(model_name)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Run 'python train.py' first to train the models.")
        return

    # Step 3: Build features and predict
    print("\n  [3/5] Building features and predicting...")
    for game in games:
        features = build_game_features(game)
        if features is None:
            game["model_prob"] = 0.5  # default if features fail
            logger.warning("Using default 0.5 prob for %s @ %s",
                           game["away_team"], game["home_team"])
        else:
            game["model_prob"] = predict_win_prob(model, features)
        game["model_name"] = model_name

    # Step 4: Fetch odds and match
    print("\n  [4/5] Fetching live odds...")
    odds = fetch_live_odds()
    if not odds:
        print("  WARNING: Could not fetch odds. Cannot calculate EV.")
        return
    games_with_odds = match_odds_to_games(odds, games)
    if not games_with_odds:
        print("  No games matched with odds. Exiting.")
        return
    print(f"        Matched odds for {len(games_with_odds)} games.")

    # Step 5: Calculate EV and filter
    print("\n  [5/5] Calculating EV and filtering picks...")
    picks = filter_positive_ev(games_with_odds)

    # Display picks
    output = format_picks(picks)
    print(output)

    # Save to database
    if picks:
        save_predictions(picks)
        print(f"  Saved {len(picks)} picks to database.\n")
    else:
        print("  No +EV picks to save.\n")


# ---------------------------------------------------------------------------
# Phase 2: Grade yesterday's picks
# ---------------------------------------------------------------------------

def run_grading():
    """Phase 2: Fetch yesterday's results and grade pending predictions."""
    print(f"\n{'='*60}")
    print(f"  MLB BETTING MODEL — Grading ({date.today()})")
    print(f"{'='*60}")

    print("\n  Fetching yesterday's results...")
    results = get_yesterdays_results()
    if not results:
        print("  No results found for yesterday. Exiting.")
        return
    print(f"  Found results for {len(results)} games.")

    print("  Grading pending predictions...")
    grade_predictions(results)

    # Show updated stats
    show_stats()


# ---------------------------------------------------------------------------
# Show lifetime stats
# ---------------------------------------------------------------------------

def show_stats():
    """Display lifetime performance statistics."""
    stats = get_roi_stats()
    output = format_stats(stats)
    print(output)


# ---------------------------------------------------------------------------
# Incremental retrain
# ---------------------------------------------------------------------------

def run_retrain(force: bool = False):
    """Incremental model retrain entry point (used by CLI and scheduler)."""
    from train import run_incremental_retrain
    run_incremental_retrain(force=force)


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
        choices=["xgboost", "logistic_regression"],
        help="Which model to use for predictions (default: xgboost).",
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Start the daily scheduler.",
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

    # Determine action
    if args.train:
        from train import main as train_main
        train_main()
    elif args.retrain:
        run_retrain(force=args.force)
    elif args.run_now:
        run_predictions(model_name=args.model)
    elif args.grade:
        run_grading()
    elif args.stats:
        show_stats()
    elif args.schedule:
        run_scheduler()
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