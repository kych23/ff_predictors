# Serving / Research DB Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. Several steps are **destructive on Supabase** and gated — do not run them until the local research DB is verified byte-for-byte.

**Goal:** Get FantasyForecast under the Supabase free-tier 500 MB cap permanently by splitting storage: a tiny **serving** DB (Supabase) holds only what the API and live draft/weekly serving read (~30 MB); a local **research** DB (native Postgres) holds the full raw/feature/label/backtest history (~640 MB) that only the pipeline, training, and benchmarks touch.

**Architecture:** Two SQLAlchemy engines chosen by *role*. Pipeline/training/benchmark code writes and reads the **research** engine (`RESEARCH_DATABASE_URL`, local Postgres). The API + `board.py`/`start.py` serving reads use the **serving** engine (`DATABASE_URL`, Supabase). A new `scripts/publish_serving.py` copies the serving subset (players, adp, forward projections, forward weekly projections, current-season weekly_features) research → serving after each pipeline run. `draft_sessions` stays serving-only (written by the API). `RESEARCH_DATABASE_URL` falls back to `DATABASE_URL` when unset, so existing single-DB dev/test setups and the whole pytest suite keep working unchanged.

**Tech Stack:** local Postgres 15 (Homebrew `postgresql@15`), SQLAlchemy 2.x, `pg_dump`/`pg_restore` for the one-time migration, existing psycopg2.

## Global Constraints

- Python via venv: `venv/bin/pytest`, `venv/bin/python` from repo root.
- Full suite green before/after every commit: `venv/bin/pytest`.
- **Backward compatible:** with only `DATABASE_URL` set (no `RESEARCH_DATABASE_URL`), both engines resolve to the same DB — the current behavior. Tests never require two DBs.
- **Serving table set** (the only tables serving reads): `players`, `adp`, `projections`, `weekly_projections`, `weekly_features` (current season only), `draft_sessions`, plus `ingest_snapshots` (one latest row, for provenance). Everything else is research-only.
- **No destructive Supabase operation** (`TRUNCATE`/`DROP`/`DELETE`) runs until Task 4 verifies the local research DB has equal-or-greater row counts for every migrated table.
- Snapshot reproducibility invariant holds: the research DB becomes the source of truth for snapshot-stamped data; nothing snapshot-stamped is deleted, only relocated.
- ADP wall untouched (no `src/projection`/`src/features` imports added).
- Commit style: terse lowercase, trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Dual-engine `session.py` (role-based, backward compatible)

Grow one engine into two, selected by role, defaulting to research so the pipeline (the bulk of call sites) needs no change. `RESEARCH_DATABASE_URL` falls back to `DATABASE_URL`.

**Files:**
- Modify: `src/db/session.py`
- Test: `tests/test_db_session_roles.py`

**Interfaces:**
- Produces:
  - `get_engine(role: str = "research")` — cached engine per role.
  - `session_scope(role: str = "research")` — role-aware scope (default research keeps pipeline call sites unchanged).
  - `serving_session_scope()` — convenience = `session_scope("serving")`.
  - `SessionLocal` — the API/serving factory, bound to the **serving** engine (get_db uses it).
  - `latest_snapshot_id()` / `resolve_snapshot()` — query the **research** engine (snapshots live there).

- [ ] **Step 1: Write the failing test**

```python
"""Role-based engine selection with single-DB fallback."""
import importlib

def test_defaults_to_single_db_when_research_url_unset(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/serving")
    monkeypatch.delenv("RESEARCH_DATABASE_URL", raising=False)
    import src.db.session as sess; importlib.reload(sess)
    assert sess._resolve_url("research") == sess._resolve_url("serving")

def test_distinct_urls_when_both_set(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/serving")
    monkeypatch.setenv("RESEARCH_DATABASE_URL", "postgresql://u@h/research")
    import src.db.session as sess; importlib.reload(sess)
    assert sess._resolve_url("research").endswith("/research")
    assert sess._resolve_url("serving").endswith("/serving")
```

- [ ] **Step 2: Run to confirm failure**

Run: `venv/bin/pytest tests/test_db_session_roles.py -v`
Expected: FAIL — `_resolve_url` missing.

- [ ] **Step 3: Implement dual engines**

Replace the single-engine block with a per-role cache. `_resolve_url("serving")` → `DATABASE_URL`; `_resolve_url("research")` → `RESEARCH_DATABASE_URL or DATABASE_URL`. `get_engine(role)` caches one engine per role. `SessionLocal` binds serving on first use. `session_scope(role="research")` yields from the role's factory. `latest_snapshot_id`/`resolve_snapshot` use `session_scope("research")`.

- [ ] **Step 4: Run tests + full suite**

Run: `venv/bin/pytest tests/test_db_session_roles.py && venv/bin/pytest`
Expected: PASS (fallback keeps every existing test on one DB).

- [ ] **Step 5: Commit**

```bash
git add src/db/session.py tests/test_db_session_roles.py
git commit -m "add role-based dual db engines with single-db fallback"
```

---

### Task 2: Point serving readers at the serving engine

The few serve-time readers must use the serving DB explicitly; pipeline code keeps the research default.

**Files:**
- Modify: `src/recommender/board.py` (`load_board` → `serving_session_scope`), `scripts/start.py` (serving reads), `api/deps.py` (`get_db` already uses `SessionLocal` = serving — verify), `api/draft_service.py` (`get_cached_board` uses `load_board` — inherits).
- Test: `tests/test_board_uses_serving.py`

**Interfaces:**
- Consumes: `serving_session_scope` (Task 1).
- Produces: no signature changes; only the engine the readers bind to.

- [ ] **Step 1: Write the failing test (monkeypatch scope, assert serving role used)**

```python
def test_load_board_uses_serving_scope(monkeypatch):
    import src.recommender.board as board
    called = {}
    from contextlib import contextmanager
    @contextmanager
    def fake_serving():
        called["serving"] = True
        raise RuntimeError("stop-after-scope")  # we only assert the scope choice
    monkeypatch.setattr(board, "serving_session_scope", fake_serving, raising=False)
    ...
```

(Full test asserts `load_board` enters `serving_session_scope`, not the research default.)

- [ ] **Step 2: Run to confirm failure, then repoint `board.py`**

`load_board` imports and uses `serving_session_scope()` instead of `session_scope()`. `start.py`'s serving reads (`_load_projections`, `_load_dvp_ranks`, player names) wrap in `serving_session_scope()`.

- [ ] **Step 3: Run suite**

Run: `venv/bin/pytest`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/recommender/board.py scripts/start.py tests/test_board_uses_serving.py
git commit -m "route serving reads through the serving engine"
```

---

### Task 3: `publish_serving.py` — copy the serving subset research → serving

After a pipeline run, push only the serve-time rows to Supabase. Idempotent per table (delete-then-insert the published subset).

**Files:**
- Create: `scripts/publish_serving.py`, `src/db/publish.py`
- Test: `tests/test_publish.py`

**Interfaces:**
- Consumes: research + serving engines, existing upserts (`src/db/upsert_data.py`).
- Produces: `publish_serving(*, season: int | None = None) -> dict[str, int]` — copies `players`, `adp` (current season), `projections` (forward rows for `season`), `weekly_projections` (forward rows), `weekly_features` (rows for `season`), and the latest `ingest_snapshots` row; returns per-table counts.

- [ ] **Step 1: Write the failing test (two in-memory SQLite DBs)**

Stand up a research SQLite and a serving SQLite (portable tables only); seed research with players + forward projections; assert `publish_serving` copies them to serving and leaves research untouched.

- [ ] **Step 2: Run to confirm failure, then implement `publish.py`**

Read the subset from `session_scope("research")`, write via `serving_session_scope()` using the existing upsert helpers. "Forward rows" = projections whose `model_version` matches the current config hash and whose season has no labels (serving season); simplest: publish rows for the given `--season` (default: max projection season).

- [ ] **Step 3: Run tests + full suite**

Run: `venv/bin/pytest tests/test_publish.py && venv/bin/pytest`
Expected: PASS.

- [ ] **Step 4: Wire into the pipeline**

Append to `scripts/run_pipeline.sh` a final step: `venv/bin/python scripts/publish_serving.py`. Document it.

- [ ] **Step 5: Commit**

```bash
git add scripts/publish_serving.py src/db/publish.py tests/test_publish.py scripts/run_pipeline.sh
git commit -m "add publish step: copy serving subset research->serving"
```

---

### Task 4: One-time data migration (Supabase → local research) + verify

Stand up the local research DB, load the full current Supabase contents into it, and verify row counts before anything is deleted from Supabase.

**Files:**
- Create: `scripts/migrate_to_local_research.sh` (documented, re-runnable)
- Modify: `.env` (add `RESEARCH_DATABASE_URL`), `.env.local.example` (names only)

- [ ] **Step 1: Create the local database**

```bash
/opt/homebrew/opt/postgresql@15/bin/pg_ctl -D /opt/homebrew/var/postgresql@15 start 2>/dev/null || brew services start postgresql@15
createdb ff_research
```

Add to `.env`: `RESEARCH_DATABASE_URL=postgresql://localhost:5432/ff_research`.

- [ ] **Step 2: Dump Supabase, restore into local**

```bash
pg_dump "$DATABASE_URL" --no-owner --no-privileges -Fc -f /tmp/ff_supabase.dump
pg_restore --no-owner --no-privileges -d "postgresql://localhost:5432/ff_research" /tmp/ff_supabase.dump
```

- [ ] **Step 3: Verify row counts match (GATE for Task 5)**

Run a script that compares `SELECT count(*)` for every public table between `DATABASE_URL` and `RESEARCH_DATABASE_URL`. **Every research count must be ≥ the Supabase count.** Print a table; abort if any mismatch.

- [ ] **Step 4: Run the full pipeline suite against local research to prove it works**

Point the pipeline at research (default role) and run the benchmark: `venv/bin/python scripts/run_benchmark.py` — Tier-1/2/3 gates must reproduce the Supabase numbers (same data). This proves the local DB is a faithful, usable copy.

- [ ] **Step 5: Commit the migration tooling**

```bash
git add scripts/migrate_to_local_research.sh .env.local.example
git commit -m "add supabase->local research migration script"
```

---

### Task 5: Trim Supabase to the serving subset (DESTRUCTIVE — gated on Task 4)

Reclaim Supabase space: delete research-only tables' data, keep only the serving subset. Runs **only after** Task 4's row-count gate passes and a fresh publish repopulates the serving tables.

**Files:**
- Create: `scripts/trim_supabase_to_serving.py` (explicit, prints a dry-run diff first)

- [ ] **Step 1: Publish the serving subset to Supabase**

Run: `venv/bin/python scripts/publish_serving.py`
Expected: players/adp/projections(forward)/weekly_projections(forward)/weekly_features(season) present on Supabase.

- [ ] **Step 2: Dry-run the trim**

`scripts/trim_supabase_to_serving.py --dry-run` lists, per research-only table (`weekly_stats_raw`, `weekly_labels`, `season_features`, `season_labels`, `player_id_map`, non-current `weekly_features`, OOF `projections`/`weekly_projections`), the row count that would be deleted and the projected reclaimed size. **Show this output and stop for explicit go-ahead.**

- [ ] **Step 3: Execute the trim (after go-ahead)**

`scripts/trim_supabase_to_serving.py` deletes research-only data, then `VACUUM FULL` each trimmed table to reclaim physical space. Re-check `pg_database_size` < 500 MB.

- [ ] **Step 4: Smoke the serving path against trimmed Supabase**

Start the API, hit `/players?season=2026`, create a session, get recommendations — all must work against the trimmed serving DB. Run `scripts/mock_draft.py --season 2024 --position 3` (reads serving) to confirm.

- [ ] **Step 5: Commit**

```bash
git add scripts/trim_supabase_to_serving.py
git commit -m "add supabase trim-to-serving script"
```

---

### Task 6: Docs

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `.claude/skills/ff-predictors/SKILL.md`

- [ ] **Step 1: Document the two-DB model**

README + CLAUDE.md: research DB (local Postgres, `RESEARCH_DATABASE_URL`, full history, pipeline target) vs serving DB (Supabase, `DATABASE_URL`, serving subset, API target); `publish_serving.py` after every pipeline run; single-DB fallback for fresh setups. Update the SKILL.md gotchas.

- [ ] **Step 2: Commit**

```bash
git add README.md CLAUDE.md .claude/skills/ff-predictors/SKILL.md
git commit -m "document serving/research db split"
```

---

## Self-Review Notes

- **Backward compatibility is load-bearing:** the whole pytest suite and any single-DB dev setup keep working because `RESEARCH_DATABASE_URL` defaults to `DATABASE_URL`. Tests never need two DBs.
- **Destructive-op safety:** Task 5 cannot run before Task 4's row-count gate; local is a verified superset before Supabase loses anything, and a `pg_dump` file remains at `/tmp/ff_supabase.dump`.
- **Serving-set correctness:** derived from actual reads — `board.py` (players/adp/projections), `start.py` (players/projections/weekly_projections/weekly_features), `api` (draft_sessions). `weekly_features` is published current-season-only (start.py's DvP display) to keep it tiny.
- **Provenance:** the latest `ingest_snapshots` row is published so the API's `latest_snapshot_id()` still stamps sessions; full snapshot history stays in research.
