"""Tests for database.py — schema, migration, CRUD, and ROI stats."""

import sqlite3
import pytest
import tempfile
import os
from unittest.mock import patch

import database


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary database path and patch DB_PATH to use it."""
    db_file = tmp_path / "test_bets.db"
    with patch("database.DB_PATH", db_file):
        database.init_db()
        yield db_file


def get_conn(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema and migration
# ---------------------------------------------------------------------------

class TestSchema:
    def test_init_creates_table(self, tmp_db):
        with get_conn(tmp_db) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        assert any(t["name"] == "predictions" for t in tables)

    def test_all_columns_present(self, tmp_db):
        with get_conn(tmp_db) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
        required = {
            "id", "date", "home_team", "away_team", "pick", "pick_side",
            "model_prob", "implied_prob", "ev", "edge", "units", "odds",
            "status", "result", "profit", "model_name", "home_pitcher",
            "away_pitcher", "created_at",
        }
        assert required.issubset(cols)

    def test_migration_adds_missing_columns(self, tmp_path):
        """Simulate an old DB without the new columns; migration should add them."""
        db_file = tmp_path / "old.db"
        old_schema = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            pick TEXT NOT NULL,
            model_prob REAL NOT NULL,
            implied_prob REAL NOT NULL,
            ev REAL NOT NULL,
            edge REAL NOT NULL,
            units REAL NOT NULL,
            odds INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending'
        );
        """
        with sqlite3.connect(str(db_file)) as conn:
            conn.executescript(old_schema)

        with patch("database.DB_PATH", db_file):
            database.init_db()  # should migrate

        with get_conn(db_file) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
        assert "pick_side" in cols
        assert "home_pitcher" in cols
        assert "away_pitcher" in cols

    def test_migration_is_idempotent(self, tmp_db):
        """Running init_db twice should not error."""
        with patch("database.DB_PATH", tmp_db):
            database.init_db()  # second call
            database.init_db()  # third call — should be fine


# ---------------------------------------------------------------------------
# save_predictions
# ---------------------------------------------------------------------------

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


class TestSavePredictions:
    def test_saves_single_pick(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([_make_pick()])
        with get_conn(tmp_db) as conn:
            rows = conn.execute("SELECT * FROM predictions").fetchall()
        assert len(rows) == 1

    def test_saved_pick_fields(self, tmp_db):
        pick = _make_pick()
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([pick])
        with get_conn(tmp_db) as conn:
            row = conn.execute("SELECT * FROM predictions").fetchone()
        assert row["pick_side"] == "Home"
        assert row["home_pitcher"] == "Cole"
        assert row["away_pitcher"] == "Sale"
        assert row["status"] == "Pending"

    def test_saves_multiple_picks(self, tmp_db):
        picks = [
            _make_pick(pick="NYY", pick_side="Home"),
            _make_pick(pick="BOS", pick_side="Away", home_team="NYY", away_team="BOS"),
        ]
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions(picks)
        with get_conn(tmp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert count == 2

    def test_empty_list_is_noop(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([])
        with get_conn(tmp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# grade_predictions
# ---------------------------------------------------------------------------

class TestGradePredictions:
    def test_win_graded_correctly(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([_make_pick(pick="NYY", units=2, odds=-115)])
            results = {"BOS @ NYY": {"home_score": 5, "away_score": 3, "winner": "NYY"}}
            database.grade_predictions(results)

        with get_conn(tmp_db) as conn:
            row = conn.execute("SELECT * FROM predictions").fetchone()
        assert row["status"] == "Win"
        # profit = 2 * (100/115) ≈ 1.739
        assert row["profit"] == pytest.approx(2 * (100 / 115), abs=1e-3)

    def test_loss_graded_correctly(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([_make_pick(pick="NYY", units=2, odds=-115)])
            results = {"BOS @ NYY": {"home_score": 2, "away_score": 5, "winner": "BOS"}}
            database.grade_predictions(results)

        with get_conn(tmp_db) as conn:
            row = conn.execute("SELECT * FROM predictions").fetchone()
        assert row["status"] == "Loss"
        assert row["profit"] == pytest.approx(-2.0, abs=1e-4)

    def test_positive_odds_win_profit(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([_make_pick(pick="BOS", units=1, odds=130)])
            results = {"BOS @ NYY": {"home_score": 3, "away_score": 4, "winner": "BOS"}}
            database.grade_predictions(results)

        with get_conn(tmp_db) as conn:
            row = conn.execute("SELECT * FROM predictions").fetchone()
        assert row["status"] == "Win"
        assert row["profit"] == pytest.approx(1.30, abs=1e-4)

    def test_unmatched_game_stays_pending(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([_make_pick()])
            database.grade_predictions({})  # no results

        with get_conn(tmp_db) as conn:
            row = conn.execute("SELECT status FROM predictions").fetchone()
        assert row["status"] == "Pending"

    def test_no_pending_is_noop(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            # Grade with no picks saved — should not error
            database.grade_predictions({"BOS @ NYY": {"winner": "NYY"}})


# ---------------------------------------------------------------------------
# get_roi_stats
# ---------------------------------------------------------------------------

class TestGetROIStats:
    def _setup_graded(self, tmp_db, picks_and_results):
        with patch("database.DB_PATH", tmp_db):
            for pick, result in picks_and_results:
                database.save_predictions([pick])
                database.grade_predictions(result)

    def test_empty_returns_zeros(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            stats = database.get_roi_stats()
        assert stats["total_bets"] == 0
        assert stats["roi_pct"] == 0.0

    def test_win_rate_calculation(self, tmp_db):
        pairs = [
            (_make_pick(pick="NYY", units=1, odds=-110),
             {"BOS @ NYY": {"winner": "NYY"}}),
            (_make_pick(pick="NYY", units=1, odds=-110),
             {"BOS @ NYY": {"winner": "BOS"}}),
        ]
        self._setup_graded(tmp_db, pairs)
        with patch("database.DB_PATH", tmp_db):
            stats = database.get_roi_stats()
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 50.0

    def test_roi_calculation(self, tmp_db):
        # 1 unit bet at +100, wins → profit = 1.0
        pairs = [
            (_make_pick(pick="NYY", units=1, odds=100),
             {"BOS @ NYY": {"winner": "NYY"}}),
        ]
        self._setup_graded(tmp_db, pairs)
        with patch("database.DB_PATH", tmp_db):
            stats = database.get_roi_stats()
        assert stats["total_profit"] == pytest.approx(1.0, abs=1e-4)
        assert stats["roi_pct"] == pytest.approx(100.0, abs=1e-2)

    def test_brier_score_present(self, tmp_db):
        pairs = [
            (_make_pick(pick="NYY", model_prob=0.60, units=1, odds=-110),
             {"BOS @ NYY": {"winner": "NYY"}}),
        ]
        self._setup_graded(tmp_db, pairs)
        with patch("database.DB_PATH", tmp_db):
            stats = database.get_roi_stats()
        assert stats["brier_score"] is not None
        # Brier = (0.60 - 1.0)^2 = 0.16
        assert stats["brier_score"] == pytest.approx(0.16, abs=1e-4)

    def test_pending_counted_separately(self, tmp_db):
        with patch("database.DB_PATH", tmp_db):
            database.save_predictions([_make_pick()])  # left as Pending
            stats = database.get_roi_stats()
        assert stats["pending"] == 1
        assert stats["total_bets"] == 0  # pending not counted in totals
