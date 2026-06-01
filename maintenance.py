"""Maintenance helpers: health checks, duplicate removal, and DB backups.

This module backs the `/daily-check` slash command and the `--health-check` /
`--dedupe` CLI flags. Everything here is local-only (no network) and designed to
be reversible: `dedupe_predictions()` snapshots the database before touching it
and records every removed row to a committed JSON file.

Duplicate identity = one pick per (date, home_team, away_team, bet_type,
model_name) — mirrors the `ux_predictions_game` unique index in database.py.
The `pick` itself is intentionally excluded so a re-run that flips the pick is
still treated as the same slot rather than a second row.
"""

import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path

import database

logger = logging.getLogger(__name__)


# --- Paths (derived from the live DB location so tests stay isolated) ----------

def _db_path() -> Path:
    """Current database path. Read at call time so patched DB_PATH is honored."""
    return Path(str(database.DB_PATH))


def _backups_dir() -> Path:
    return _db_path().parent / "backups"


def _maintenance_dir() -> Path:
    return _db_path().parent / "maintenance"


def _data_dir() -> Path:
    return _db_path().parent / "docs" / "data"


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


# --- Backups -------------------------------------------------------------------

def backup_database() -> Path:
    """Write a consistent, standalone snapshot of the DB to backups/.

    Uses SQLite's online backup API so the copy is correct even in WAL mode.
    Returns the backup file path. Backups are gitignored (in-session undo); git
    history is the durable record.
    """
    src_path = _db_path()
    backups = _backups_dir()
    backups.mkdir(parents=True, exist_ok=True)
    dest = backups / f"{src_path.stem}_{_utc_stamp()}.db"

    src = sqlite3.connect(str(src_path))
    try:
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    logger.info("Database backed up to %s", dest)
    return dest


# --- Duplicate detection & removal ---------------------------------------------

def find_duplicates() -> list[dict]:
    """Return duplicate groups (more than one row sharing the unique key).

    Each group is a dict with: ``key`` (the identifying tuple as a dict),
    ``kept_id``, ``removed_ids``, and ``removed_rows`` (full row dicts, suitable
    for re-insertion). The kept row is a graded row if any, else the lowest id —
    so dedup never discards a settled Win/Loss in favor of a Pending duplicate.
    """
    with database.get_connection() as conn:
        # status='Pending' sorts graded rows (0) ahead of pending (1); id ASC
        # breaks ties. The first row per group is the keeper.
        rows = conn.execute(
            "SELECT * FROM predictions "
            "ORDER BY date, home_team, away_team, "
            "COALESCE(bet_type, 'moneyline'), COALESCE(model_name, ''), "
            "(status = 'Pending') ASC, id ASC"
        ).fetchall()

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        r = dict(row)
        key = (
            r["date"],
            r["home_team"],
            r["away_team"],
            r.get("bet_type") or "moneyline",
            r.get("model_name") or "",
        )
        groups.setdefault(key, []).append(r)

    duplicates = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        kept, removed = members[0], members[1:]
        duplicates.append({
            "key": {
                "date": key[0],
                "home_team": key[1],
                "away_team": key[2],
                "bet_type": key[3],
                "model_name": key[4],
            },
            "kept_id": kept["id"],
            "removed_ids": [r["id"] for r in removed],
            "removed_rows": removed,
        })
    return duplicates


def dedupe_predictions(reexport: bool = True) -> dict:
    """Remove duplicate predictions, keeping one row per unique key.

    Reversible by design: the DB is snapshotted first and every removed row is
    written to maintenance/removed_rows_<ts>.json before deletion. Afterwards the
    unique index is (re)created so future re-runs can't reintroduce duplicates,
    and (optionally) the dashboard JSON is re-exported from the cleaned DB.

    Returns a summary dict.
    """
    duplicates = find_duplicates()
    if not duplicates:
        # Still ensure the index exists now that the table is clean.
        _ensure_unique_index()
        logger.info("No duplicate predictions found.")
        return {
            "removed_count": 0,
            "group_count": 0,
            "backup_path": None,
            "record_path": None,
        }

    backup_path = backup_database()

    removed_rows = [r for g in duplicates for r in g["removed_rows"]]
    removed_ids = [rid for g in duplicates for rid in g["removed_ids"]]

    self_dir = _maintenance_dir()
    self_dir.mkdir(parents=True, exist_ok=True)
    record_path = self_dir / f"removed_rows_{_utc_stamp()}.json"
    record = {
        "removed_at": datetime.utcnow().isoformat() + "Z",
        "backup": str(backup_path),
        "groups": [
            {"key": g["key"], "kept_id": g["kept_id"], "removed_ids": g["removed_ids"]}
            for g in duplicates
        ],
        "removed_rows": removed_rows,
    }
    record_path.write_text(json.dumps(record, indent=2, default=str))

    with database.get_connection() as conn:
        conn.executemany(
            "DELETE FROM predictions WHERE id = ?",
            [(rid,) for rid in removed_ids],
        )
    logger.info(
        "Removed %d duplicate row(s) across %d group(s).",
        len(removed_ids), len(duplicates),
    )

    _ensure_unique_index()

    if reexport:
        # Imported lazily to avoid a circular import (main imports maintenance).
        from main import export_dashboard_data
        export_dashboard_data()

    return {
        "removed_count": len(removed_ids),
        "group_count": len(duplicates),
        "backup_path": str(backup_path),
        "record_path": str(record_path),
    }


def _ensure_unique_index() -> None:
    """Create the unique index if the table is now free of duplicates."""
    with database.get_connection() as conn:
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_predictions_game "
                "ON predictions(date, home_team, away_team, bet_type, model_name)"
            )
        except sqlite3.IntegrityError:
            logger.warning(
                "Duplicates still present — unique index not created. "
                "Re-run dedupe after investigating."
            )


# --- Health check --------------------------------------------------------------

def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def health_check() -> dict:
    """Read-only daily sanity check. Returns a report dict (see issues list)."""
    today = date.today().isoformat()
    report: dict = {"today": today, "issues": []}

    with database.get_connection() as conn:
        today_count = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE date = ?", (today,)
        ).fetchone()["c"]
        pending_past = conn.execute(
            "SELECT date, COUNT(*) AS c FROM predictions "
            "WHERE status = 'Pending' AND date < ? GROUP BY date ORDER BY date",
            (today,),
        ).fetchall()

    report["today_predicted"] = today_count > 0
    report["today_pick_count"] = today_count

    ungraded = {r["date"]: r["c"] for r in pending_past}
    report["ungraded_past"] = sum(ungraded.values())
    report["ungraded_past_dates"] = ungraded

    # Duplicates
    dups = find_duplicates()
    report["duplicate_groups"] = len(dups)
    report["duplicate_rows"] = sum(len(g["removed_ids"]) for g in dups)

    # Export freshness (stats.json last_updated should be today, UTC)
    stats = _read_json(_data_dir() / "stats.json")
    last_updated = (stats or {}).get("last_updated")
    report["export_last_updated"] = last_updated
    today_utc = datetime.utcnow().date().isoformat()
    report["stale_export"] = not (last_updated or "").startswith(today_utc)

    # JSON vs DB consistency (mirror the export filters in main.export_dashboard_data)
    picks_today = _read_json(_data_dir() / "picks_today.json")
    picks_history = _read_json(_data_dir() / "picks_history.json")
    consistency: dict = {}
    with database.get_connection() as conn:
        db_today_ml = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions "
            "WHERE date = ? AND COALESCE(bet_type, 'moneyline') = 'moneyline'",
            (today,),
        ).fetchone()["c"]
        db_history = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions "
            "WHERE NOT (date = ? AND COALESCE(status, 'Pending') = 'Pending')",
            (today,),
        ).fetchone()["c"]
    consistency["today_db"] = db_today_ml
    consistency["today_json"] = len(picks_today) if isinstance(picks_today, list) else None
    consistency["today_match"] = (
        consistency["today_json"] == db_today_ml
        if isinstance(picks_today, list) else None
    )
    consistency["history_db"] = db_history
    consistency["history_json"] = len(picks_history) if isinstance(picks_history, list) else None
    consistency["history_match"] = (
        consistency["history_json"] == db_history
        if isinstance(picks_history, list) else None
    )
    report["json_db_consistency"] = consistency

    # Roll up issues
    issues = report["issues"]
    if report["ungraded_past"]:
        issues.append(
            f"{report['ungraded_past']} ungraded pick(s) from past dates "
            f"({', '.join(ungraded)}) — run grading."
        )
    if report["duplicate_groups"]:
        issues.append(
            f"{report['duplicate_rows']} duplicate row(s) in "
            f"{report['duplicate_groups']} group(s) — run --dedupe."
        )
    if report["stale_export"]:
        issues.append(
            f"Dashboard export looks stale (stats.json last_updated="
            f"{last_updated!r}, expected {today_utc})."
        )
    if consistency["today_match"] is False or consistency["history_match"] is False:
        issues.append("Dashboard JSON counts don't match the database — re-export.")

    report["ok"] = not issues
    return report


def format_health_report(report: dict) -> str:
    """Render a health_check() report as human-readable text."""
    lines = [
        "=" * 60,
        f"  HEALTH CHECK — {report['today']}",
        "=" * 60,
    ]
    status = "OK — no issues found" if report["ok"] else f"{len(report['issues'])} issue(s) found"
    lines.append(f"  Status: {status}")
    lines.append("")
    lines.append(f"  Today's picks in DB:     {report['today_pick_count']}"
                 f"{'' if report['today_predicted'] else '  (no games / not run yet)'}")
    lines.append(f"  Ungraded past picks:     {report['ungraded_past']}")
    lines.append(f"  Duplicate rows:          {report['duplicate_rows']} "
                 f"in {report['duplicate_groups']} group(s)")
    lines.append(f"  Export last_updated:     {report['export_last_updated']}"
                 f"{'  (STALE)' if report['stale_export'] else ''}")
    c = report["json_db_consistency"]
    lines.append(f"  picks_today  json/db:    {c['today_json']}/{c['today_db']}")
    lines.append(f"  picks_history json/db:   {c['history_json']}/{c['history_db']}")
    if report["issues"]:
        lines.append("")
        lines.append("  Issues:")
        for issue in report["issues"]:
            lines.append(f"    - {issue}")
    lines.append("=" * 60)
    return "\n".join(lines)
