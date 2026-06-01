---
description: Daily health check — verify picks & grades updated cleanly, remove duplicate rows, log every change reversibly.
---

# Daily Check

Run a daily maintenance pass on the MLB betting model. Goals, in order:

1. **No glitches / everything updated** — confirm today's picks and yesterday's
   grades actually landed and that the dashboard JSON matches the database.
2. **No duplicates** — detect and remove duplicate prediction rows.
3. **Logged & reversible** — record every change and keep it easy to undo.

All steps below are local-only and safe except where noted. Do **not** push
unless I explicitly ask. Work on the current branch.

## Steps

### 1. Health check (read-only)

Run:

```bash
python main.py --health-check
```

Read the report carefully. It flags: whether today has picks, ungraded picks
from past dates, duplicate rows, a stale dashboard export, and any
JSON-vs-database count mismatch. If `Status: OK — no issues found` and there are
no duplicates, skip to step 5 and just log the clean run.

Optionally, cross-check that the scheduled GitHub Actions actually ran: use the
GitHub MCP tools to look at the most recent **Daily Predictions** and **Daily
Grade** workflow runs / commits on the default branch and note any failures. If
you can't reach GitHub, say so and move on.

### 2. Fix staleness (only if flagged, and only if you can)

- If **ungraded past picks** are reported, grade them:
  ```bash
  python main.py --grade
  ```
- If **today was never predicted** (no picks, and it isn't an off-day), run:
  ```bash
  python main.py --run-now
  ```

Both hit external APIs (MLB Stats / The Odds API) and need secrets. In the web
sandbox the network may be blocked — if either command fails for that reason,
**do not force it**: report the gap in your summary and continue. Dedup and the
health check below still work offline.

### 3. Remove duplicates (only if the report shows any)

```bash
python main.py --dedupe
```

This backs up the database to `backups/` first, removes duplicate rows (keeping a
graded row over a Pending one, else the earliest), writes the removed rows to
`maintenance/removed_rows_<timestamp>.json`, re-exports the dashboard JSON, and
prints a summary. Note the backup path and removed count for the log.

### 4. Re-verify

```bash
python main.py --health-check
```

Confirm duplicates are now `0` and consistency lines match.

### 5. Log the run + commit

Prepend a dated entry to `MAINTENANCE_LOG.md` (newest first, under the
`<!-- New entries ... -->` marker) summarizing:

- UTC date/time of the run
- Health-check status before (and key flags)
- Whether grading / prediction was re-run, or why it was skipped
- Duplicates removed (count + groups), backup path, and removed-rows file
- Health-check status after

Then stage and commit the changes:

```bash
git add mlb_bets.db docs/data/ maintenance/ MAINTENANCE_LOG.md
git status   # confirm backups/ is NOT staged (it is gitignored)
git commit -m "chore: daily maintenance check $(date -u +%Y-%m-%d)"
```

Do not push unless I ask.

### 6. Summary

Report back briefly: what was stale (if anything), how many duplicates were
removed, which files changed, and exactly how to undo — `git revert` of this
commit, or restore the timestamped `backups/*.db` snapshot.
