"""SQLite database layer for storing and grading predictions."""

import sqlite3
import logging
from datetime import date
from typing import Optional

import pandas as pd

from config import DB_PATH, SUPABASE_URL, SUPABASE_SERVICE_KEY

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    pick TEXT NOT NULL,
    pick_side TEXT,
    model_prob REAL NOT NULL,
    implied_prob REAL NOT NULL,
    ev REAL NOT NULL,
    edge REAL NOT NULL,
    units REAL NOT NULL,
    odds INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'Pending',
    result TEXT,
    profit REAL,
    model_name TEXT,
    home_pitcher TEXT,
    away_pitcher TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raw model vs market probability for EVERY game with odds (not just picks).
-- Graded outcomes feed calibration.update_blend_weight(), which learns how
-- much to shrink the model toward the market. Logging the full slate avoids
-- the adverse-selection bias a picks-only sample would have.
CREATE TABLE IF NOT EXISTS model_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    raw_model_prob REAL NOT NULL,
    market_prob REAL NOT NULL,
    home_odds INTEGER,
    away_odds INTEGER,
    model_name TEXT,
    home_win INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, home_team, away_team)
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _migrate_db(conn: sqlite3.Connection):
    """Add columns introduced after initial schema without dropping existing data."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    new_columns = [
        # Original migrations
        ("pick_side", "TEXT"),
        ("home_pitcher", "TEXT"),
        ("away_pitcher", "TEXT"),
        # Totals & confidence columns
        ("bet_type", "TEXT DEFAULT 'moneyline'"),
        ("listed_total", "REAL"),
        ("predicted_total", "REAL"),
        ("predicted_home_runs", "REAL"),
        ("predicted_away_runs", "REAL"),
        ("total_delta", "REAL"),
        ("confidence", "INTEGER"),
        # Pre-blend model probability, kept for calibration audits
        ("raw_model_prob", "REAL"),
        # Final score, captured at grading time. Without these the totals model
        # can never be calibrated: TOTALS_SIGMA and the run-scale constants in
        # score.py have to be fit against real residuals, and the residuals
        # were being thrown away.
        ("actual_home_score", "INTEGER"),
        ("actual_away_score", "INTEGER"),
        ("actual_total", "REAL"),
    ]
    for col, col_type in new_columns:
        if col not in existing:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
            logger.info("Migration: added column '%s' to predictions.", col)


def init_db():
    """Create the predictions table if it doesn't exist, then migrate."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        _migrate_db(conn)
    logger.info("Database initialized at %s", DB_PATH)


def save_predictions(picks: list[dict], bet_type: str = "moneyline"):
    """Insert today's picks into the database with 'Pending' status.

    Each pick dict should have at minimum: date, home_team, away_team, pick,
    pick_side, model_prob, implied_prob, ev, edge, units, odds, model_name,
    home_pitcher, away_pitcher. Totals picks additionally carry listed_total,
    predicted_total, predicted_home_runs, predicted_away_runs, total_delta.
    """
    if not picks:
        logger.info("No picks to save.")
        return

    sql = """
    INSERT INTO predictions
        (date, home_team, away_team, pick, pick_side, model_prob, implied_prob,
         ev, edge, units, odds, status, model_name, home_pitcher, away_pitcher,
         bet_type, listed_total, predicted_total, predicted_home_runs,
         predicted_away_runs, total_delta, confidence, raw_model_prob)
    VALUES
        (:date, :home_team, :away_team, :pick, :pick_side, :model_prob, :implied_prob,
         :ev, :edge, :units, :odds, 'Pending', :model_name, :home_pitcher, :away_pitcher,
         :bet_type, :listed_total, :predicted_total, :predicted_home_runs,
         :predicted_away_runs, :total_delta, :confidence, :raw_model_prob)
    """
    normalized = []
    for p in picks:
        row = dict(p)
        row.setdefault("bet_type", bet_type)
        row.setdefault("listed_total", None)
        row.setdefault("predicted_total", None)
        row.setdefault("predicted_home_runs", None)
        row.setdefault("predicted_away_runs", None)
        row.setdefault("total_delta", None)
        row.setdefault("confidence", None)
        row.setdefault("raw_model_prob", None)
        normalized.append(row)

    with get_connection() as conn:
        conn.executemany(sql, normalized)
    logger.info("Saved %d predictions to database.", len(picks))


def grade_predictions(results: dict[str, dict], for_date: Optional[str] = None):
    """Grade pending predictions using actual game results.

    Args:
        results: Dict mapping game keys ("away @ home") to
                 {"home_score": int, "away_score": int, "winner": str}.
        for_date: If given, only grade pending predictions on this date
                  (YYYY-MM-DD). Prevents a same-matchup result from one day
                  grading a still-pending pick on a different day.
    """
    with get_connection() as conn:
        sql = ("SELECT id, home_team, away_team, pick, units, odds, "
               "bet_type, listed_total "
               "FROM predictions WHERE status = 'Pending'")
        params: list = []
        if for_date:
            sql += " AND date = ?"
            params.append(for_date)
        pending = conn.execute(sql, params).fetchall()

        if not pending:
            logger.info("No pending predictions to grade%s.",
                        f" for {for_date}" if for_date else "")
            return

        graded = 0
        for row in pending:
            game_key = f"{row['away_team']} @ {row['home_team']}"
            if game_key not in results:
                logger.warning("No result found for %s, skipping.", game_key)
                continue

            result = results[game_key]
            pick  = row["pick"]
            units = row["units"]
            odds  = row["odds"]
            home_score = result.get("home_score", 0)
            away_score = result.get("away_score", 0)
            actual_total = home_score + away_score

            # Determine winner depending on bet type
            bet_type = row["bet_type"] if "bet_type" in row.keys() else "moneyline"
            if bet_type == "totals":
                listed = row["listed_total"] if "listed_total" in row.keys() else None
                if listed is None:
                    actual_winner = None
                elif actual_total == listed:
                    # Whole-number line landing exactly on the total is a PUSH:
                    # the stake is returned. Grading it as "Under" (the old
                    # behaviour) booked a full-unit loss on every push and
                    # understated the totals record.
                    actual_winner = "Push"
                else:
                    actual_winner = "Over" if actual_total > listed else "Under"
            else:
                actual_winner = result["winner"]

            if actual_winner is None:
                continue

            if actual_winner == "Push":
                status, profit = "Push", 0.0
            elif actual_winner == pick:
                status = "Win"
                if odds > 0:
                    profit = units * (odds / 100)
                else:
                    profit = units * (100 / abs(odds))
            else:
                status = "Loss"
                profit = -units

            conn.execute(
                "UPDATE predictions SET status = ?, result = ?, profit = ?, "
                "actual_home_score = ?, actual_away_score = ?, actual_total = ? "
                "WHERE id = ?",
                (status, actual_winner, profit, home_score, away_score,
                 actual_total, row["id"]),
            )
            graded += 1

        logger.info("Graded %d predictions.", graded)


def save_model_log(rows: list[dict]):
    """Upsert raw model vs market probabilities for a slate of games.

    One row per (date, home_team, away_team). Re-running predictions for the
    same day refreshes the probabilities, but never touches a graded outcome.
    """
    if not rows:
        return
    sql = """
    INSERT INTO model_log
        (date, home_team, away_team, raw_model_prob, market_prob,
         home_odds, away_odds, model_name)
    VALUES
        (:date, :home_team, :away_team, :raw_model_prob, :market_prob,
         :home_odds, :away_odds, :model_name)
    ON CONFLICT(date, home_team, away_team) DO UPDATE SET
        raw_model_prob = excluded.raw_model_prob,
        market_prob    = excluded.market_prob,
        home_odds      = excluded.home_odds,
        away_odds      = excluded.away_odds,
        model_name     = excluded.model_name
    WHERE model_log.home_win IS NULL
    """
    with get_connection() as conn:
        conn.executemany(sql, rows)
    logger.info("Saved %d rows to model_log.", len(rows))


def grade_model_log(results: dict[str, dict], for_date: Optional[str] = None):
    """Attach actual outcomes to ungraded model_log rows.

    Args:
        results: Dict mapping "AWAY @ HOME" to {"home_score", "away_score", "winner"}.
        for_date: Only grade rows on this date (YYYY-MM-DD) when given.
    """
    with get_connection() as conn:
        sql = "SELECT id, home_team, away_team FROM model_log WHERE home_win IS NULL"
        params: list = []
        if for_date:
            sql += " AND date = ?"
            params.append(for_date)
        pending = conn.execute(sql, params).fetchall()

        graded = 0
        for row in pending:
            game_key = f"{row['away_team']} @ {row['home_team']}"
            result = results.get(game_key)
            if result is None:
                continue
            home_win = 1 if result["winner"] == row["home_team"] else 0
            conn.execute(
                "UPDATE model_log SET home_win = ? WHERE id = ?",
                (home_win, row["id"]),
            )
            graded += 1
        if pending:
            logger.info("Graded %d / %d model_log rows%s.", graded, len(pending),
                        f" for {for_date}" if for_date else "")


def get_graded_model_log() -> list[dict]:
    """Return all graded model_log rows for blend-weight calibration."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT raw_model_prob, market_prob, home_win
               FROM model_log WHERE home_win IS NOT NULL"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_model_log_dates_pending() -> list[str]:
    """Distinct dates with ungraded model_log rows (for grading backfill)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT date FROM model_log WHERE home_win IS NULL"
        ).fetchall()
    return [r["date"] for r in rows]


def get_pending_dates() -> list[str]:
    """Return distinct dates that have pending predictions."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT date FROM predictions WHERE status = 'Pending'"
        ).fetchall()
    return [r["date"] for r in rows]


def get_roi_stats(since: Optional[str] = None,
                  bet_type: Optional[str] = None) -> dict:
    """Calculate ROI statistics.

    Args:
        since: Only count picks on or after this ISO date.
        bet_type: Restrict to one market ("moneyline" or "totals"). None (the
            default) covers every market.

    Pushes are excluded from the win/loss record and from the ROI denominator
    (the stake is returned), but are reported in their own ``pushes`` count.

    Returns dict with: total_bets, wins, losses, pushes, pending,
    total_units_wagered, total_profit, roi_pct, brier_score, win_rate.
    """
    # A NULL bet_type predates the totals migration and means moneyline.
    if bet_type == "moneyline":
        type_clause = " AND (bet_type = 'moneyline' OR bet_type IS NULL)"
        type_params: list = []
    elif bet_type:
        type_clause = " AND bet_type = ?"
        type_params = [bet_type]
    else:
        type_clause = ""
        type_params = []

    with get_connection() as conn:
        where = "WHERE status != 'Pending'" + type_clause
        params = list(type_params)
        if since:
            where += " AND date >= ?"
            params.append(since)

        rows = conn.execute(
            f"SELECT model_prob, status, profit, units FROM predictions {where}",
            params,
        ).fetchall()

        # The pending count must use the SAME bet_type and date filters as the
        # graded query. It previously ignored bet_type entirely, so a
        # moneyline-only stat block reported the pending count for every market.
        pending_sql = "SELECT COUNT(*) as cnt FROM predictions WHERE status = 'Pending'" + type_clause
        pending_params = list(type_params)
        if since:
            pending_sql += " AND date >= ?"
            pending_params.append(since)
        pending_count = conn.execute(pending_sql, pending_params).fetchone()["cnt"]

    if not rows:
        return {
            "total_bets": 0, "wins": 0, "losses": 0, "pushes": 0,
            "pending": pending_count,
            "total_units_wagered": 0, "total_profit": 0.0,
            "roi_pct": 0.0, "brier_score": None, "win_rate": 0.0,
        }

    wins = sum(1 for r in rows if r["status"] == "Win")
    losses = sum(1 for r in rows if r["status"] == "Loss")
    pushes = sum(1 for r in rows if r["status"] == "Push")
    decided = [r for r in rows if r["status"] in ("Win", "Loss")]
    total_units = sum(r["units"] for r in decided)
    total_profit = sum(r["profit"] for r in rows if r["profit"] is not None)

    # Brier score: mean squared error of predicted prob vs actual outcome.
    # Pushes have no binary outcome, so they are excluded.
    brier_score = None
    if decided:
        brier_sum = sum(
            (r["model_prob"] - (1.0 if r["status"] == "Win" else 0.0)) ** 2
            for r in decided
        )
        brier_score = round(brier_sum / len(decided), 4)

    return {
        "total_bets": len(rows),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pending": pending_count,
        "total_units_wagered": total_units,
        "total_profit": round(total_profit, 2),
        "roi_pct": round((total_profit / total_units) * 100, 2) if total_units > 0 else 0.0,
        "brier_score": brier_score,
        "win_rate": round((wins / len(decided)) * 100, 2) if decided else 0.0,
    }


def get_recent_predictions(days: int = 7) -> pd.DataFrame:
    """Return recent predictions as a DataFrame."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY date DESC, id DESC LIMIT ?",
            conn,
            params=[days * 20],  # rough upper bound
        )
    return df


def upload_picks_to_supabase(today_picks: list[dict], history: list[dict]) -> bool:
    """Upload picks JSON to Supabase private Storage bucket.

    Returns True on success, False if Supabase is not configured or upload fails.
    Both files are upserted so repeated runs overwrite stale data.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False
    try:
        import json
        import math
        from supabase import create_client

        def _clean(obj):
            if isinstance(obj, float) and not math.isfinite(obj):
                return None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        client  = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        bucket  = client.storage.from_("picks-data")
        options = {"content-type": "application/json", "upsert": "true"}

        for name, data in [("picks_today", today_picks), ("picks_history", history)]:
            payload = json.dumps(_clean(data), indent=2).encode()
            bucket.upload(f"{name}.json", payload, file_options=options)

        logger.info("Uploaded picks to Supabase Storage.")
        return True
    except Exception as exc:
        logger.warning("Supabase Storage upload failed: %s", exc)
        return False


def get_all_predictions() -> list[dict]:
    """Return all predictions as JSON-serializable dicts, newest first."""
    import math
    with get_connection() as conn:
        df = pd.read_sql_query(
            """SELECT date, home_team, away_team, pick, pick_side,
                      model_prob, implied_prob, edge, ev, units, odds,
                      status, result, profit, home_pitcher, away_pitcher,
                      bet_type, listed_total, predicted_total,
                      predicted_home_runs, predicted_away_runs, total_delta,
                      confidence, raw_model_prob,
                      actual_home_score, actual_away_score, actual_total
               FROM predictions
               ORDER BY date DESC, id DESC""",
            conn,
        )
    float_cols = [
        "model_prob", "implied_prob", "edge", "ev", "units", "profit",
        "listed_total", "predicted_total", "predicted_home_runs",
        "predicted_away_runs", "total_delta", "raw_model_prob", "actual_total",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    # to_dict() on float64 columns silently keeps NaN even after .where().
    # Fix: clean each value explicitly after converting to records.
    records = df.to_dict(orient="records")
    return [
        {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
         for k, v in row.items()}
        for row in records
    ]
