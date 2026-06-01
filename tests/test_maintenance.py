"""Tests for maintenance.py — dedup, backups, source-level dup protection, health."""

import json
import sqlite3
from datetime import date, timedelta
from unittest.mock import patch

import pytest

import database
import maintenance


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary initialized DB with DB_PATH patched everywhere it's read."""
    db_file = tmp_path / "test_bets.db"
    with patch("database.DB_PATH", db_file):
        database.init_db()
        yield db_file


def get_conn(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _make_pick(**overrides):
    base = {
        "date": "2026-04-02",
        "home_team": "NYY",
        "away_team": "BOS",
        "pick": "NYY",
        "pick_side": "Home",
        "model_prob": 0.58,
        "implied_prob": 0.52,
        "ev": 0.05,
        "edge": 0.06,
        "units": 3,
        "odds": -115,
        "model_name": "xgboost",
        "home_pitcher": "Cole",
        "away_pitcher": "Sale",
    }
    base.update(overrides)
    return base


def _drop_index(db_file):
    """Simulate a legacy DB (pre unique-index) so duplicates can be inserted."""
    with get_conn(db_file) as conn:
        conn.execute("DROP INDEX IF EXISTS ux_predictions_game")


# ---------------------------------------------------------------------------
# Source-level protection (unique index + INSERT OR IGNORE)
# ---------------------------------------------------------------------------

class TestSourceProtection:
    def test_unique_index_created(self, tmp_db):
        with get_conn(tmp_db) as conn:
            idx = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name='ux_predictions_game'"
            ).fetchall()
        assert idx, "ux_predictions_game unique index should exist after init_db"

    def test_resaving_same_pick_is_noop(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([_make_pick()])
            database.save_predictions([_make_pick()])  # re-run same day
        with get_conn(tmp_db) as conn:
            rows = conn.execute("SELECT * FROM predictions").fetchall()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Duplicate detection & removal
# ---------------------------------------------------------------------------

class TestDedup:
    def test_find_and_remove_duplicates(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            _drop_index(tmp_db)  # legacy DB without the guard
            database.save_predictions([_make_pick()])
            database.save_predictions([_make_pick()])  # duplicate slips in

            dups = maintenance.find_duplicates()
            assert len(dups) == 1
            assert len(dups[0]["removed_ids"]) == 1

            summary = maintenance.dedupe_predictions(reexport=False)
            assert summary["removed_count"] == 1
            assert summary["group_count"] == 1

        with get_conn(tmp_db) as conn:
            rows = conn.execute("SELECT * FROM predictions").fetchall()
        assert len(rows) == 1

    def test_removed_rows_recorded(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            _drop_index(tmp_db)
            database.save_predictions([_make_pick()])
            database.save_predictions([_make_pick()])
            summary = maintenance.dedupe_predictions(reexport=False)

        record = json.loads((tmp_db.parent / "maintenance" /
                             summary["record_path"].split("/")[-1]).read_text())
        assert len(record["removed_rows"]) == 1
        assert record["removed_rows"][0]["home_team"] == "NYY"

    def test_backup_created(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            _drop_index(tmp_db)
            database.save_predictions([_make_pick()])
            database.save_predictions([_make_pick()])
            summary = maintenance.dedupe_predictions(reexport=False)
        assert summary["backup_path"] is not None
        assert (tmp_db.parent / "backups").exists()
        # backup is a readable standalone SQLite file
        with get_conn(summary["backup_path"]) as conn:
            assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2

    def test_keeps_graded_over_pending(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            _drop_index(tmp_db)
            database.save_predictions([_make_pick()])
            database.save_predictions([_make_pick()])
            # Grade exactly one of the two duplicate rows (the later one).
            with database.get_connection() as conn:
                first_id = conn.execute(
                    "SELECT id FROM predictions ORDER BY id LIMIT 1"
                ).fetchone()["id"]
                conn.execute(
                    "UPDATE predictions SET status='Win', result='NYY', profit=2.6 "
                    "WHERE id != ?", (first_id,)
                )
                graded_id = conn.execute(
                    "SELECT id FROM predictions WHERE status='Win'"
                ).fetchone()["id"]

            dups = maintenance.find_duplicates()
            assert dups[0]["kept_id"] == graded_id

            maintenance.dedupe_predictions(reexport=False)
        with get_conn(tmp_db) as conn:
            rows = conn.execute("SELECT * FROM predictions").fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "Win"

    def test_dedupe_noop_when_clean(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([_make_pick()])
            summary = maintenance.dedupe_predictions(reexport=False)
        assert summary["removed_count"] == 0
        assert summary["backup_path"] is None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_flags_ungraded_past_pick(self, tmp_db):
        past = (date.today() - timedelta(days=3)).isoformat()
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([_make_pick(date=past)])
            report = maintenance.health_check()
        assert report["ungraded_past"] == 1
        assert any("ungraded" in i for i in report["issues"])
        assert report["ok"] is False

    def test_flags_stale_export(self, tmp_db):
        # No stats.json in the temp data dir → export looks stale.
        with patch("database.DB_PATH", tmp_db):
            report = maintenance.health_check()
        assert report["stale_export"] is True

    def test_flags_duplicates(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            _drop_index(tmp_db)
            database.save_predictions([_make_pick()])
            database.save_predictions([_make_pick()])
            report = maintenance.health_check()
        assert report["duplicate_groups"] == 1
        assert report["duplicate_rows"] == 1
        assert any("duplicate" in i for i in report["issues"])

    def test_report_formats(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            text = maintenance.format_health_report(maintenance.health_check())
        assert "HEALTH CHECK" in text
