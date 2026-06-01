# Maintenance Log

A dated record of manual maintenance runs (health checks and duplicate cleanups)
performed via the `/daily-check` slash command or `python main.py --health-check`
/ `--dedupe`. Every change is reversible: duplicate removals snapshot the DB to
`backups/` (gitignored, in-session undo) and record the removed rows to
`maintenance/removed_rows_*.json` (committed). Git history of `mlb_bets.db` and
`docs/data/` is the durable undo.

Newest entries first.

<!-- New entries are appended below this line by /daily-check -->

## 2026-06-01 — Initial dedupe + health tooling

- **Health check before:** 30 duplicate rows across 30 groups; no unique index;
  336 total prediction rows. (Duplicates came from days where the predict job
  ran twice — same game/pick, often with moved odds, sometimes both graded.)
- **Action:** ran `--dedupe`. Kept the earliest morning pick per game (preferring
  an already-graded row); removed the later re-run copy.
- **Removed:** 30 rows (336 → 306).
- **Stats impact (all-time):** double-counting corrected.
  - Bets 291 → 273, Wins 130 → 117, Losses 161 → 156.
  - Profit **+19.75u → −1.81u**; ROI **+4.99% → −0.50%**.
- **Backup:** `backups/mlb_bets_20260601T005913Z.db` (gitignored, in-session undo).
- **Removed-rows record:** `maintenance/removed_rows_20260601T005913Z.json` (committed).
- **Source fix:** added unique index `ux_predictions_game` + `INSERT OR IGNORE`,
  so future re-runs can't reintroduce duplicates.
- **Re-export:** `docs/data/{stats,picks_today,picks_history}.json` regenerated.
- **Undo:** `git revert` this commit, or restore the timestamped backup above.
