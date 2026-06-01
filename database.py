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
        # MLB gamePk — distinguishes the two games of a doubleheader (same date /
        # teams). Part of the uniqueness key so legitimate doubleheader picks are
        # not collapsed into one.
        ("game_id", "INTEGER"),
    ]
    for col, col_type in new_columns:
        if col not in existing:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
            logger.info("Migration: added column '%s' to predictions.", col)

    # Enforce one pick per game / bet_type / model / day at the source. Combined
    # with INSERT OR IGNORE in save_predictions(), this makes re-running a day a
    # no-op instead of inserting duplicate rows. If the table already contains
    # duplicates the index can't be created — we warn and skip rather than delete
    # silently. Run `python main.py --dedupe` to clean up (logged + reversible);
    # the index is then created inside maintenance.dedupe_predictions().
    cols_now = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    index_cols = "date, home_team, away_team, bet_type, model_name, game_id"
    if {"date", "home_team", "away_team", "bet_type", "model_name", "game_id"}.issubset(cols_now):
        # If an older index without game_id exists, drop it so the definition can
        # be upgraded (CREATE ... IF NOT EXISTS won't replace a same-named index).
        existing_idx = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='ux_predictions_game'"
        ).fetchone()
        if existing_idx and existing_idx[0] and "game_id" not in existing_idx[0]:
            conn.execute("DROP INDEX ux_predictions_game")
        try:
            conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS ux_predictions_game "
                f"ON predictions({index_cols})"
            )
        except sqlite3.Error:
            logger.warning(
                "Duplicate predictions present — unique index not created. "
                "Run 'python main.py --dedupe' to remove duplicates."
            )


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

    # INSERT OR IGNORE: once the ux_predictions_game unique index exists, a
    # re-run of the same day silently skips already-saved games instead of
    # creating duplicates. It keeps the original row (never overwrites a saved or
    # already-graded pick) — intended.
    sql = """
    INSERT OR IGNORE INTO predictions
        (date, game_id, home_team, away_team, pick, pick_side, model_prob, implied_prob,
         ev, edge, units, odds, status, model_name, home_pitcher, away_pitcher,
         bet_type, listed_total, predicted_total, predicted_home_runs,
         predicted_away_runs, total_delta, confidence)
    VALUES
        (:date, :game_id, :home_team, :away_team, :pick, :pick_side, :model_prob, :implied_prob,
         :ev, :edge, :units, :odds, 'Pending', :model_name, :home_pitcher, :away_pitcher,
         :bet_type, :listed_total, :predicted_total, :predicted_home_runs,
         :predicted_away_runs, :total_delta, :confidence)
    """
    normalized = []
    for p in picks:
        row = dict(p)
        row.setdefault("game_id", None)
        row.setdefault("bet_type", bet_type)
        row.setdefault("listed_total", None)
        row.setdefault("predicted_total", None)
        row.setdefault("predicted_home_runs", None)
        row.setdefault("predicted_away_runs", None)
        row.setdefault("total_delta", None)
        row.setdefault("confidence", None)
        normalized.append(row)

    with get_connection() as conn:
        conn.executemany(sql, normalized)
    logger.info("Saved %d predictions to database.", len(picks))


def grade_predictions(results: dict[str, dict], target_date: Optional[str] = None):
    """Grade pending predictions using actual game results.

    Args:
        results: Dict mapping game keys ("away @ home") to
                 {"home_score": int, "away_score": int, "winner": str}.
        target_date: If given (YYYY-MM-DD), only grade pending picks from that
                 date. This scopes grading when sweeping a backlog of stale
                 dates so a result from one date can't grade a same-matchup
                 pending pick on another date.
    """
    query = (
        "SELECT id, home_team, away_team, pick, units, odds, "
        "bet_type, listed_total "
        "FROM predictions WHERE status = 'Pending'"
    )
    params: list = []
    if target_date:
        query += " AND date = ?"
        params.append(target_date)
    with get_connection() as conn:
        pending = conn.execute(query, params).fetchall()

        if not pending:
            logger.info("No pending predictions to grade.")
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

            # Determine winner depending on bet type
            bet_type = row["bet_type"] if "bet_type" in row.keys() else "moneyline"
            if bet_type == "totals":
                actual_total = result.get("home_score", 0) + result.get("away_score", 0)
                listed = row["listed_total"] if "listed_total" in row.keys() else None
                if listed is not None:
                    actual_winner = "Over" if actual_total > listed else "Under"
                else:
                    actual_winner = None
            else:
                actual_winner = result["winner"]

            if actual_winner is None:
                continue

            if actual_winner == pick:
                status = "Win"
                if odds > 0:
                    profit = units * (odds / 100)
                else:
                    profit = units * (100 / abs(odds))
            else:
                status = "Loss"
                profit = -units

            conn.execute(
                "UPDATE predictions SET status = ?, result = ?, profit = ? WHERE id = ?",
                (status, actual_winner, profit, row["id"]),
            )
            graded += 1

        logger.info("Graded %d predictions.", graded)


def get_pending_dates() -> list[str]:
    """Return distinct dates that have pending predictions."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT date FROM predictions WHERE status = 'Pending'"
        ).fetchall()
    return [r["date"] for r in rows]


def get_roi_stats(since: Optional[str] = None) -> dict:
    """Calculate ROI statistics.

    Returns dict with: total_bets, wins, losses, pending, total_units_wagered,
    total_profit, roi_pct, brier_score, win_rate.
    """
    with get_connection() as conn:
        where = "WHERE status != 'Pending' AND (bet_type = 'moneyline' OR bet_type IS NULL)"
        params = []
        if since:
            where += " AND date >= ?"
            params.append(since)

        rows = conn.execute(
            f"SELECT model_prob, status, profit, units FROM predictions {where}",
            params,
        ).fetchall()

    if not rows:
        with get_connection() as conn:
            pending_sql = "SELECT COUNT(*) as cnt FROM predictions WHERE status = 'Pending' AND (bet_type = 'moneyline' OR bet_type IS NULL)"
            pending_params = []
            if since:
                pending_sql += " AND date >= ?"
                pending_params.append(since)
            pending_count = conn.execute(pending_sql, pending_params).fetchone()["cnt"]
        return {
            "total_bets": 0, "wins": 0, "losses": 0, "pending": pending_count,
            "total_units_wagered": 0, "total_profit": 0.0,
            "roi_pct": 0.0, "brier_score": None, "win_rate": 0.0,
        }

    wins = sum(1 for r in rows if r["status"] == "Win")
    losses = sum(1 for r in rows if r["status"] == "Loss")
    total_units = sum(r["units"] for r in rows)
    total_profit = sum(r["profit"] for r in rows if r["profit"] is not None)

    # Brier score: mean squared error of predicted prob vs actual outcome
    brier_sum = 0.0
    for r in rows:
        actual = 1.0 if r["status"] == "Win" else 0.0
        brier_sum += (r["model_prob"] - actual) ** 2
    brier_score = brier_sum / len(rows)

    with get_connection() as conn:
        pending_sql = "SELECT COUNT(*) as cnt FROM predictions WHERE status = 'Pending'"
        pending_params = []
        if since:
            pending_sql += " AND date >= ?"
            pending_params.append(since)
        pending_count = conn.execute(pending_sql, pending_params).fetchone()["cnt"]

    return {
        "total_bets": len(rows),
        "wins": wins,
        "losses": losses,
        "pending": pending_count,
        "total_units_wagered": total_units,
        "total_profit": round(total_profit, 2),
        "roi_pct": round((total_profit / total_units) * 100, 2) if total_units > 0 else 0.0,
        "brier_score": round(brier_score, 4),
        "win_rate": round((wins / len(rows)) * 100, 2) if rows else 0.0,
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
                      predicted_home_runs, predicted_away_runs, total_delta, confidence
               FROM predictions
               ORDER BY date DESC, id DESC""",
            conn,
        )
    float_cols = [
        "model_prob", "implied_prob", "edge", "ev", "units", "profit",
        "listed_total", "predicted_total", "predicted_home_runs",
        "predicted_away_runs", "total_delta",
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
