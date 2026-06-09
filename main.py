"""Main orchestrator — daily prediction pipeline with scheduler and CLI."""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

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
from evaluate import filter_positive_ev, filter_totals_ev, format_picks, format_stats, compute_confidence
from score import predict_game_scores
from calibration import (
    log_model_predictions,
    update_blend_weight,
    get_blend_state,
    get_blend_weight,
    is_self_tuned,
)
from config import MARKET_BLEND_WEIGHT, TOTALS_MAX_DISAGREEMENT

logger = logging.getLogger(__name__)


def post_picks_to_github_issue(picks: list[dict], totals_picks: list[dict] | None = None) -> None:
    """Create a GitHub Issue with today's picks (moneyline + totals).

    Requires GITHUB_TOKEN and GITHUB_REPOSITORY env vars (set automatically in Actions).
    Silently skips if either is missing.
    """
    import os
    token = os.getenv("GITHUB_TOKEN", "")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not token or not repo:
        return

    totals_picks = totals_picks or []
    today = date.today().isoformat()
    total_count = len(picks) + len(totals_picks)

    if total_count == 0:
        title = f"⚾ MLB Picks — No picks for {today}"
        body = "No +EV picks found today."
    else:
        title = f"⚾ MLB Picks — {len(picks)} ML + {len(totals_picks)} O/U for {today}"
        sections = []
        if picks:
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
            sections.append("\n".join(ml_rows))
        if totals_picks:
            ou_rows = ["## Totals (Over/Under) Picks",
                       "| Game | Pick | Conf | Odds | Model | Edge | EV | Units |",
                       "|------|------|:----:|-----:|------:|-----:|---:|------:|"]
            for p in totals_picks:
                matchup  = f"{p['away_team']} @ {p['home_team']}"
                odds_str = f"+{p['odds']}" if p["odds"] > 0 else str(p["odds"])
                stars    = "★" * p.get("confidence", 1)
                ou_rows.append(
                    f"| {matchup} | **{p['pick']} {p['listed_total']}** | {stars} | {odds_str} "
                    f"| {p['predicted_total']} | +{p['edge']:.1%} | {p['ev']:+.1%} | {p['units']:.1f}u |"
                )
            sections.append("\n".join(ou_rows))
        total_units = sum(p["units"] for p in picks) + sum(p["units"] for p in totals_picks)
        sections.append(f"**Total: {total_count} picks · {total_units:.1f} units wagered**")
        body = "\n\n".join(sections)

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


def post_picks_to_discord(picks: list[dict], totals_picks: list[dict] | None = None) -> None:
    """POST today's picks (moneyline + totals) to Discord via webhook.

    Silently skips if DISCORD_WEBHOOK_URL is not configured.
    """
    from config import DISCORD_WEBHOOK_URL
    if not DISCORD_WEBHOOK_URL:
        return

    totals_picks = totals_picks or []
    total_count = len(picks) + len(totals_picks)

    if total_count == 0:
        payload = {"content": "⚾ **BaseballBetBot** — No +EV picks for today."}
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
        for p in totals_picks:
            matchup = f"{p['away_team']} @ {p['home_team']}"
            odds_str = f"+{p['odds']}" if p["odds"] > 0 else str(p["odds"])
            stars = "★" * p.get("confidence", 1)
            lines.append(
                f"📊 **{p['pick']} {p['listed_total']}** ({matchup}) {stars}\n"
                f"   Odds: `{odds_str}` · Model: `{p['predicted_total']}` · "
                f"Edge: `+{p['edge']:.1%}` · Units: `{p['units']:.1f}u` · EV: `{p['ev']:+.1%}`"
            )
        lines.append(f"\n_{total_count} pick(s) today — Good luck! 🍀_")
        payload = {"content": "\n".join(lines)}

    try:
        import requests
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Posted %d pick(s) to Discord.", total_count)
    except Exception as e:
        logger.warning("Discord webhook failed: %s", e)


# ---------------------------------------------------------------------------
# Idempotency marker — lets us schedule the prediction run redundantly
# ---------------------------------------------------------------------------
#
# GitHub Actions `schedule` crons are best-effort: at busy times they are
# delayed by hours or silently dropped with no run ever created. To guarantee
# predictions post, the workflow fires several times a day. This marker makes
# those extra runs safe: once the pipeline reaches a terminal, successful state
# (picks posted, or a confirmed off day), it records today's date and later
# runs no-op. Transient failures (MLB/odds API outage, missing model) leave NO
# marker on purpose, so a subsequent scheduled run retries and recovers.

PREDICT_STATUS_FILE = Path("docs/data/predict_status.json")


def _predictions_done_today() -> bool:
    """True if the prediction pipeline already completed successfully today."""
    import json
    try:
        data = json.loads(PREDICT_STATUS_FILE.read_text())
    except (FileNotFoundError, ValueError, OSError):
        return False
    return data.get("date") == date.today().isoformat()


def _mark_predictions_done(picks_posted: int) -> None:
    """Record a terminal, successful prediction run for idempotency."""
    import json
    from datetime import datetime
    PREDICT_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREDICT_STATUS_FILE.write_text(json.dumps({
        "date": date.today().isoformat(),
        "picks_posted": picks_posted,
        "ran_at": datetime.utcnow().isoformat() + "Z",
    }, indent=2))


# ---------------------------------------------------------------------------
# Phase 1: Morning prediction run
# ---------------------------------------------------------------------------

def run_predictions(model_name: str = "xgboost", force: bool = False):
    """Phase 1: Fetch today's games, predict, fetch odds, calculate EV, save picks.

    Idempotent: if a successful run already posted today, this is a no-op unless
    ``force`` is set. Redundant scheduled runs therefore can't double-post.
    """
    print(f"\n{'='*60}")
    print(f"  MLB BETTING MODEL — Daily Predictions ({date.today()})")
    print(f"{'='*60}")

    if not force and _predictions_done_today():
        print("\n  Predictions already completed for today — skipping (idempotent).")
        logger.info("Predictions already posted today; skipping redundant run.")
        return

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
        export_dashboard_data()
        _mark_predictions_done(0)
        return
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
            game["predicted_total"] = None
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
                game["predicted_total"] = scores["predicted_total"]
            except Exception as e:
                logger.warning("Score prediction failed for %s @ %s: %s",
                               game["away_team"], game["home_team"], e)
                game["predicted_home_runs"] = None
                game["predicted_away_runs"] = None
                game["predicted_total"] = None
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

    # Log raw model vs market probability for the FULL slate. Graded outcomes
    # feed the self-tuning blend weight (see calibration.py).
    try:
        log_model_predictions(games_with_odds)
    except Exception as e:
        logger.warning("Calibration logging failed (non-fatal): %s", e)

    # Step 5: Calculate EV, attach confidence, and filter picks
    print("\n  [5/5] Calculating EV and filtering picks...")
    print(f"        Market blend weight: {get_blend_weight():.2f} "
          f"({'self-tuned' if is_self_tuned() else 'default'})")
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

    # Totals (Over/Under) picks from the score model + posted O/U lines.
    totals_picks = filter_totals_ev(games_with_odds)
    for p in totals_picks:
        p["confidence"] = compute_confidence(
            p["edge"], p["ev"],
            max_disagreement=TOTALS_MAX_DISAGREEMENT,
            weight=MARKET_BLEND_WEIGHT,
        )

    # Display picks
    print(format_picks(ml_picks))
    if totals_picks:
        print(f"  Plus {len(totals_picks)} Over/Under pick(s):")
        for p in totals_picks:
            print(f"    {p['away_team']} @ {p['home_team']}: {p['pick']} {p['listed_total']} "
                  f"(model {p['predicted_total']}, edge +{p['edge']:.1%}, {p['units']:.1f}u)")

    # Save to database
    if ml_picks:
        save_predictions(ml_picks, bet_type="moneyline")
        print(f"  Saved {len(ml_picks)} moneyline picks to database.")
    if totals_picks:
        save_predictions(totals_picks, bet_type="totals")
        print(f"  Saved {len(totals_picks)} totals picks to database.")
    if not ml_picks and not totals_picks:
        print("  No +EV picks to save.\n")

    all_picks = ml_picks + totals_picks
    export_dashboard_data()
    post_picks_to_discord(ml_picks, totals_picks)
    post_picks_to_github_issue(ml_picks, totals_picks)
    _mark_predictions_done(len(all_picks))


# ---------------------------------------------------------------------------
# Phase 2: Grade yesterday's picks
# ---------------------------------------------------------------------------

def run_grading():
    """Phase 2: Grade pending predictions.

    Grades yesterday's slate AND backfills any older still-pending dates. The
    daily run used to only fetch yesterday's results, so any pick missed on its
    day (a failed run, or a game not yet final) stayed Pending forever. We now
    retry every date that still has pending picks, date-scoped so results from
    one day can't grade a same-matchup pick on another.
    """
    print(f"\n{'='*60}")
    print(f"  MLB BETTING MODEL — Grading ({date.today()})")
    print(f"{'='*60}")

    from database import get_pending_dates, get_model_log_dates_pending, grade_model_log

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    pending_dates = get_pending_dates()
    pending_log_dates = get_model_log_dates_pending()
    dates_to_grade = sorted(set(pending_dates) | set(pending_log_dates) | {yesterday})
    print(f"\n  Grading dates: {', '.join(dates_to_grade)}")

    graded_any = False
    for d in dates_to_grade:
        try:
            results = get_yesterdays_results(date.fromisoformat(d))
        except Exception as e:
            print(f"  ERROR: results unavailable for {d} — {e}")
            continue
        if not results:
            print(f"  {d}: no final results yet.")
            continue
        print(f"  {d}: grading against {len(results)} final games.")
        grade_predictions(results, for_date=d)
        grade_model_log(results, for_date=d)
        graded_any = True

    if not graded_any:
        print("  Nothing graded this run.")

    # Self-tuning step: refit the market blend weight on all graded games.
    try:
        state = update_blend_weight()
        if state:
            print(f"\n  Blend weight self-tuned to {state['weight']:.2f} "
                  f"on {state['n_games']} graded games "
                  f"(log loss {state['log_loss']:.5f}).")
        else:
            print("\n  Blend weight: not enough graded games yet — using default.")
    except Exception as e:
        logger.warning("Blend weight update failed (non-fatal): %s", e)

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
    blend_state = get_blend_state()
    stats = {
        "ytd": {**get_roi_stats(since=ytd_since), "since": ytd_since},
        "all_time": get_roi_stats(),
        "model": {
            "blend_weight": get_blend_weight(),
            "self_tuned": is_self_tuned(),
            "calibration_games": (blend_state or {}).get("n_games", 0),
            "calibration_updated": (blend_state or {}).get("updated_at"),
        },
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
    today_totals = [p for p in today_picks if p.get("bet_type") == "totals"]

    upload_picks_to_supabase(today_ml, history_for_export)
    (out / "picks_today.json").write_text(json.dumps(_sanitize_json(today_ml), indent=2))
    (out / "totals_today.json").write_text(json.dumps(_sanitize_json(today_totals), indent=2))

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
        help="With --retrain: wipe caches and rebuild from scratch. "
             "With --run-now: re-run predictions even if already posted today.",
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

    # Guard: --force is only meaningful with --retrain or --run-now
    if args.force and not (args.retrain or args.run_now):
        parser.error("--force requires --retrain or --run-now")

    # Determine action
    if args.train:
        from train import run_incremental_retrain
        run_incremental_retrain(force=True, tune=args.tune)
    elif args.retrain:
        run_retrain(force=args.force, tune=args.tune)
    elif args.run_now:
        run_predictions(model_name=args.model, force=args.force)
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